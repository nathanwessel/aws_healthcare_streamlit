"""
AWS Glue PySpark job: Bronze -> Silver - CMS Provider Information
=================================================================

Purpose
-------
Convert CMS Nursing Home Provider Information CSV snapshots from the Bronze
layer into a clean, typed Parquet dataset in the Silver layer.

Current Bronze example:
    s3://nw-healthcare-project/bronze/Nursing_Homes_data/
        NH_ProviderInfo_Oct2024.csv

Default Silver output:
    s3://nw-healthcare-project/silver/provider_info/

Design decisions
----------------
* Future-ready: finds every file matching NH_ProviderInfo_<Month><Year>.csv
  directly under the Bronze source prefix.
* Reads each CSV independently, then unions by column name. This is more
  resilient to future CMS schema changes than reading many CSVs as one file set.
* Keeps EVERY source column.
* Standardizes headers to UPPER_SNAKE_CASE.
* Special-case mapping:
      CMS Certification Number (CCN) -> CCN
* Casts all known columns according to the October 2024 CMS data dictionary:
      Text    -> STRING
      Y/N     -> STRING (values kept as Y/N)
      Date    -> DATE
      Numeric -> DOUBLE
  CMS only labels numeric fields as "Numeric"; it does not distinguish integer
  from decimal, so DOUBLE is used consistently for dictionary Numeric fields.
* Converts blank/NA/N/A/Data Not Available to NULL for numeric/date columns.
  Text columns keep legitimate text values and only blank strings become NULL.
* Does NOT remove duplicates.
* Adds:
      SOURCE_FILE_NAME
      SNAPSHOT_DATE       (first day of month derived from the filename)
      SILVER_PROCESSED_AT
* Writes Parquet with overwrite mode.
* Does NOT partition yet.
* Does NOT coalesce to one output file.
* Does NOT create/update Glue Data Catalog tables.

Recommended Glue runtime
------------------------
This script intentionally uses only standard AWS Glue / PySpark / boto3
functionality. A modern Glue Spark runtime is recommended.
"""

import re
import sys
from datetime import datetime

import boto3
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# =============================================================================
# 1. Glue / Spark setup
# =============================================================================

required_args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(required_args["JOB_NAME"], required_args)


# =============================================================================
# 2. Default paths and optional Glue parameters
# =============================================================================

DEFAULT_SOURCE_PREFIX = (
    "s3://nw-healthcare-project/bronze/Nursing_Homes_data/"
)

DEFAULT_TARGET_PATH = (
    "s3://nw-healthcare-project/silver/provider_info/"
)

# Match the current naming convention and future months, for example:
#   NH_ProviderInfo_Oct2024.csv
#   NH_ProviderInfo_Nov2024.csv
FILE_NAME_PATTERN = re.compile(
    r"^NH_ProviderInfo_[A-Za-z]+[0-9]{4}\.csv$",
    re.IGNORECASE,
)


def get_optional_arg(name: str, default: str) -> str:
    """
    Read an optional Glue job argument without making it required.

    Example Glue parameters:
        --SOURCE_PREFIX s3://some-bucket/bronze/Nursing_Homes_data/
        --TARGET_PATH   s3://some-bucket/silver/provider_info/
    """
    flag = f"--{name}"

    if flag not in sys.argv:
        return default

    index = sys.argv.index(flag)

    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for Glue argument {flag}")

    return sys.argv[index + 1]


SOURCE_PREFIX = get_optional_arg(
    "SOURCE_PREFIX",
    DEFAULT_SOURCE_PREFIX,
).rstrip("/") + "/"

TARGET_PATH = get_optional_arg(
    "TARGET_PATH",
    DEFAULT_TARGET_PATH,
).rstrip("/") + "/"


# =============================================================================
# 3. CMS data-dictionary type mapping
# =============================================================================
#
# These column names are the UPPER_SNAKE_CASE forms of the October 2024 CMS
# Provider Information variables.
#
# IMPORTANT:
# The CMS dictionary provides broad types: Text, Y/N, Date, and Numeric.
# It does not distinguish integers from decimals, so all dictionary Numeric
# fields are represented as DOUBLE in Silver.
#
# If CMS adds a future column not listed below, the job preserves that new
# column as STRING and prints a warning. This avoids silently dropping data.

TEXT_COLUMNS = {
    "CCN",
    "PROVIDER_NAME",
    "PROVIDER_ADDRESS",
    "CITY_TOWN",
    "STATE",
    "COUNTY_PARISH",
    "OWNERSHIP_TYPE",
    "PROVIDER_TYPE",
    "LEGAL_BUSINESS_NAME",
    "AFFILIATED_ENTITY_NAME",
    "SPECIAL_FOCUS_STATUS",
    "WITH_A_RESIDENT_AND_FAMILY_COUNCIL",
    "AUTOMATIC_SPRINKLER_SYSTEMS_IN_ALL_REQUIRED_AREAS",
    "LOCATION",
}

YES_NO_COLUMNS = {
    "PROVIDER_RESIDES_IN_HOSPITAL",
    "CONTINUING_CARE_RETIREMENT_COMMUNITY",
    "ABUSE_ICON",
    "MOST_RECENT_HEALTH_INSPECTION_MORE_THAN_2_YEARS_AGO",
    "PROVIDER_CHANGED_OWNERSHIP_IN_LAST_12_MONTHS",
}

DATE_COLUMNS = {
    "DATE_FIRST_APPROVED_TO_PROVIDE_MEDICARE_AND_MEDICAID_SERVICES",
    "RATING_CYCLE_1_STANDARD_SURVEY_HEALTH_DATE",
    "RATING_CYCLE_2_STANDARD_HEALTH_SURVEY_DATE",
    "RATING_CYCLE_3_STANDARD_HEALTH_SURVEY_DATE",
    "PROCESSING_DATE",
}

NUMERIC_COLUMNS = {
    "ZIP_CODE",
    "TELEPHONE_NUMBER",
    "PROVIDER_SSA_COUNTY_CODE",
    "NUMBER_OF_CERTIFIED_BEDS",
    "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
    "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY_FOOTNOTE",
    "AFFILIATED_ENTITY_ID",
    "OVERALL_RATING",
    "OVERALL_RATING_FOOTNOTE",
    "HEALTH_INSPECTION_RATING",
    "HEALTH_INSPECTION_RATING_FOOTNOTE",
    "QM_RATING",
    "QM_RATING_FOOTNOTE",
    "LONG_STAY_QM_RATING",
    "LONG_STAY_QM_RATING_FOOTNOTE",
    "SHORT_STAY_QM_RATING",
    "SHORT_STAY_QM_RATING_FOOTNOTE",
    "STAFFING_RATING",
    "STAFFING_RATING_FOOTNOTE",
    "REPORTED_STAFFING_FOOTNOTE",
    "PHYSICAL_THERAPIST_STAFFING_FOOTNOTE",
    "REPORTED_NURSE_AIDE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "REPORTED_LPN_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "REPORTED_RN_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "REPORTED_LICENSED_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "TOTAL_NUMBER_OF_NURSE_STAFF_HOURS_PER_RESIDENT_PER_DAY_ON_THE_WEEKEND",
    "REGISTERED_NURSE_HOURS_PER_RESIDENT_PER_DAY_ON_THE_WEEKEND",
    "REPORTED_PHYSICAL_THERAPIST_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "TOTAL_NURSING_STAFF_TURNOVER",
    "TOTAL_NURSING_STAFF_TURNOVER_FOOTNOTE",
    "REGISTERED_NURSE_TURNOVER",
    "REGISTERED_NURSE_TURNOVER_FOOTNOTE",
    "NUMBER_OF_ADMINISTRATORS_WHO_HAVE_LEFT_THE_NURSING_HOME",
    "ADMINISTRATOR_TURNOVER_FOOTNOTE",
    "NURSING_CASE_MIX_INDEX",
    "NURSING_CASE_MIX_INDEX_RATIO",
    "CASE_MIX_NURSE_AIDE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "CASE_MIX_LPN_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "CASE_MIX_RN_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "CASE_MIX_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "CASE_MIX_WEEKEND_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "ADJUSTED_NURSE_AIDE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "ADJUSTED_LPN_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "ADJUSTED_RN_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "ADJUSTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "ADJUSTED_WEEKEND_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "RATING_CYCLE_1_TOTAL_NUMBER_OF_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_1_NUMBER_OF_STANDARD_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_1_NUMBER_OF_COMPLAINT_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_1_HEALTH_DEFICIENCY_SCORE",
    "RATING_CYCLE_1_NUMBER_OF_HEALTH_REVISITS",
    "RATING_CYCLE_1_HEALTH_REVISIT_SCORE",
    "RATING_CYCLE_1_TOTAL_HEALTH_SCORE",
    "RATING_CYCLE_2_TOTAL_NUMBER_OF_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_2_NUMBER_OF_STANDARD_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_2_NUMBER_OF_COMPLAINT_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_2_HEALTH_DEFICIENCY_SCORE",
    "RATING_CYCLE_2_NUMBER_OF_HEALTH_REVISITS",
    "RATING_CYCLE_2_HEALTH_REVISIT_SCORE",
    "RATING_CYCLE_2_TOTAL_HEALTH_SCORE",
    "RATING_CYCLE_3_TOTAL_NUMBER_OF_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_3_NUMBER_OF_STANDARD_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_3_NUMBER_OF_COMPLAINT_HEALTH_DEFICIENCIES",
    "RATING_CYCLE_3_HEALTH_DEFICIENCY_SCORE",
    "RATING_CYCLE_3_NUMBER_OF_HEALTH_REVISITS",
    "RATING_CYCLE_3_HEALTH_REVISIT_SCORE",
    "RATING_CYCLE_3_TOTAL_HEALTH_SCORE",
    "TOTAL_WEIGHTED_HEALTH_SURVEY_SCORE",
    "NUMBER_OF_FACILITY_REPORTED_INCIDENTS",
    "NUMBER_OF_SUBSTANTIATED_COMPLAINTS",
    "NUMBER_OF_CITATIONS_FROM_INFECTION_CONTROL_INSPECTIONS",
    "NUMBER_OF_FINES",
    "TOTAL_AMOUNT_OF_FINES_IN_DOLLARS",
    "NUMBER_OF_PAYMENT_DENIALS",
    "TOTAL_NUMBER_OF_PENALTIES",
    "LATITUDE",
    "LONGITUDE",
    "GEOCODING_FOOTNOTE",
}

DICTIONARY_COLUMNS = (
    TEXT_COLUMNS
    | YES_NO_COLUMNS
    | DATE_COLUMNS
    | NUMERIC_COLUMNS
)


# =============================================================================
# 4. General helper functions
# =============================================================================

NULL_TOKENS_FOR_TYPED_FIELDS = {
    "",
    "NA",
    "N/A",
    "DATA NOT AVAILABLE",
}


def parse_s3_uri(s3_uri: str):
    """Split s3://bucket/key into (bucket, key)."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(
            f"Expected an s3:// URI, received: {s3_uri}"
        )

    without_scheme = s3_uri[5:]
    parts = without_scheme.split("/", 1)

    bucket = parts[0]
    key = parts[1] if len(parts) == 2 else ""

    return bucket, key


def list_source_files(source_prefix: str):
    """
    Find matching Provider Information CSVs directly under SOURCE_PREFIX.

    We list individual objects rather than reading a wildcard all at once so
    that each CSV can be normalized independently before the union. This helps
    if CMS adds/removes/reorders columns in a later monthly release.
    """
    bucket, prefix = parse_s3_uri(source_prefix)

    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    matching_files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Only process files directly in this folder, not nested folders.
            relative_key = key[len(prefix):]

            if "/" in relative_key:
                continue

            if FILE_NAME_PATTERN.match(relative_key):
                matching_files.append(
                    f"s3://{bucket}/{key}"
                )

    matching_files.sort()

    if not matching_files:
        raise FileNotFoundError(
            "No Provider Information CSVs were found matching "
            f"{FILE_NAME_PATTERN.pattern} under {source_prefix}"
        )

    return matching_files


def standardize_column_name(column_name: str) -> str:
    """
    Convert a CMS display-style header to UPPER_SNAKE_CASE.

    Example:
        "Average Number of Residents per Day"
            -> "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY"

        "City/Town"
            -> "CITY_TOWN"
    """
    cleaned = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        column_name.strip(),
    )
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").upper()

    # Deliberate shorter identifier agreed on for the project.
    if cleaned == "CMS_CERTIFICATION_NUMBER_CCN":
        return "CCN"

    return cleaned


def standardize_dataframe_columns(df: DataFrame) -> DataFrame:
    """
    Rename every source column and detect accidental naming collisions.
    """
    standardized_names = [
        standardize_column_name(column)
        for column in df.columns
    ]

    if len(standardized_names) != len(set(standardized_names)):
        raise ValueError(
            "Two or more source columns collapse to the same "
            "UPPER_SNAKE_CASE name. Source columns: "
            f"{df.columns}"
        )

    for old_name, new_name in zip(df.columns, standardized_names):
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)

    return df


def snapshot_date_from_file_name(file_name: str):
    """
    Parse the snapshot month from the CMS filename.

    Examples:
        NH_ProviderInfo_Oct2024.csv -> 2024-10-01
        NH_ProviderInfo_Nov2025.csv -> 2025-11-01
    """
    match = re.search(
        r"_([A-Za-z]+)([0-9]{4})\.csv$",
        file_name,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Could not derive snapshot month from filename: {file_name}"
        )

    month_year = f"{match.group(1)}{match.group(2)}"

    for date_format in ("%b%Y", "%B%Y"):
        try:
            parsed = datetime.strptime(
                month_year,
                date_format,
            )
            return parsed.date().replace(day=1)
        except ValueError:
            pass

    raise ValueError(
        f"Could not parse month/year from filename: {file_name}"
    )


def union_allow_missing_columns(dataframes):
    """
    Union DataFrames by column name while preserving new/missing CMS columns.

    This avoids requiring all monthly CSV snapshots to have identical schemas.
    Missing raw source columns are added as STRING NULLs before the union.
    """
    if not dataframes:
        raise ValueError("No DataFrames were supplied for union.")

    all_columns = []

    for df in dataframes:
        for column in df.columns:
            if column not in all_columns:
                all_columns.append(column)

    aligned_dataframes = []

    for df in dataframes:
        missing_columns = [
            column
            for column in all_columns
            if column not in df.columns
        ]

        for column in missing_columns:
            df = df.withColumn(
                column,
                F.lit(None).cast("string"),
            )

        aligned_dataframes.append(
            df.select(*all_columns)
        )

    result = aligned_dataframes[0]

    for df in aligned_dataframes[1:]:
        result = result.unionByName(df)

    return result


def trim_text(column_name: str):
    """
    For normal Text columns, only blank strings become NULL.
    Values such as 'Data Not Available' remain legitimate text.
    """
    value = F.trim(F.col(column_name))

    return F.when(
        value == "",
        F.lit(None).cast("string"),
    ).otherwise(value)


def clean_yes_no(column_name: str):
    """
    Preserve CMS Y/N fields as strings, per the project decision.
    """
    value = F.upper(F.trim(F.col(column_name)))

    return F.when(
        value == "",
        F.lit(None).cast("string"),
    ).otherwise(value)


def is_typed_null_token(value_column):
    """
    Return a Spark boolean expression identifying configured null tokens.

    This is applied only to Numeric and Date fields.
    """
    return F.upper(F.trim(value_column)).isin(
        *NULL_TOKENS_FOR_TYPED_FIELDS
    )


def clean_numeric(column_name: str):
    """
    Convert a dictionary Numeric field to DOUBLE.

    Common formatting characters are stripped before casting so values such as
    '1,234.5', '$123.45', or '52.1%' can still become numeric.
    """
    raw = F.col(column_name).cast("string")

    cleaned = F.trim(raw)
    cleaned = F.regexp_replace(cleaned, ",", "")
    cleaned = F.regexp_replace(cleaned, r"\$", "")
    cleaned = F.regexp_replace(cleaned, "%", "")

    return F.when(
        is_typed_null_token(raw),
        F.lit(None).cast("double"),
    ).otherwise(
        cleaned.cast("double")
    )


def clean_date(column_name: str):
    """
    Convert a dictionary Date field to DATE.

    CMS files commonly use MM/dd/yyyy, but a few common alternatives are
    supported to make the Silver job tolerant of formatting changes.
    """
    raw = F.col(column_name).cast("string")
    value = F.trim(raw)

    parsed_date = F.coalesce(
        F.to_date(value, "MM/dd/yyyy"),
        F.to_date(value, "M/d/yyyy"),
        F.to_date(value, "yyyy-MM-dd"),
        F.to_date(value, "MM-dd-yyyy"),
    )

    return F.when(
        is_typed_null_token(raw),
        F.lit(None).cast("date"),
    ).otherwise(parsed_date)


def apply_dictionary_types(df: DataFrame) -> DataFrame:
    """
    Apply the CMS data-dictionary types to every recognized source column.

    Future CMS columns not yet represented in the October 2024 dictionary are
    preserved as STRING rather than dropped.
    """
    metadata_columns = {
        "SOURCE_FILE_NAME",
        "SNAPSHOT_DATE",
        "SILVER_PROCESSED_AT",
    }

    unexpected_columns = sorted(
        set(df.columns)
        - DICTIONARY_COLUMNS
        - metadata_columns
    )

    if unexpected_columns:
        print(
            "[WARNING] Columns not found in the October 2024 type map "
            "will be preserved as STRING:"
        )
        for column in unexpected_columns:
            print(f"  - {column}")

    for column in df.columns:
        if column in TEXT_COLUMNS:
            df = df.withColumn(
                column,
                trim_text(column),
            )

        elif column in YES_NO_COLUMNS:
            df = df.withColumn(
                column,
                clean_yes_no(column),
            )

        elif column in DATE_COLUMNS:
            df = df.withColumn(
                column,
                clean_date(column),
            )

        elif column in NUMERIC_COLUMNS:
            df = df.withColumn(
                column,
                clean_numeric(column),
            )

    return df


# =============================================================================
# 5. Read each Bronze snapshot independently
# =============================================================================

source_files = list_source_files(SOURCE_PREFIX)

print(
    f"[INFO] Found {len(source_files)} Provider Information snapshot file(s)."
)

monthly_dataframes = []

for source_uri in source_files:
    source_file_name = source_uri.rsplit("/", 1)[-1]
    snapshot_date = snapshot_date_from_file_name(
        source_file_name
    )

    print(
        f"[READ] {source_uri} "
        f"(SNAPSHOT_DATE={snapshot_date})"
    )

    # inferSchema=false deliberately loads raw Bronze values as strings.
    # Silver is where we explicitly apply the CMS data-dictionary types.
    current_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("quote", '"')
        .option("escape", '"')
        .csv(source_uri)
    )

    current_df = standardize_dataframe_columns(
        current_df
    )

    reserved_columns = {
        "SOURCE_FILE_NAME",
        "SNAPSHOT_DATE",
        "SILVER_PROCESSED_AT",
    }

    collision = reserved_columns.intersection(
        current_df.columns
    )

    if collision:
        raise ValueError(
            "Source data already contains reserved Silver metadata "
            f"column(s): {sorted(collision)}"
        )

    current_df = (
        current_df
        .withColumn(
            "SOURCE_FILE_NAME",
            F.lit(source_file_name),
        )
        .withColumn(
            "SNAPSHOT_DATE",
            F.lit(snapshot_date).cast("date"),
        )
    )

    monthly_dataframes.append(current_df)


# =============================================================================
# 6. Union snapshots and clean/type the Silver dataset
# =============================================================================

silver_df = union_allow_missing_columns(
    monthly_dataframes
)

silver_df = (
    silver_df
    .withColumn(
        "SILVER_PROCESSED_AT",
        F.current_timestamp(),
    )
)

silver_df = apply_dictionary_types(
    silver_df
)


# =============================================================================
# 7. Helpful job logging
# =============================================================================

row_count = silver_df.count()

print(f"[INFO] Silver row count: {row_count}")
print(f"[INFO] Silver column count: {len(silver_df.columns)}")
print("[INFO] Final Silver schema:")
silver_df.printSchema()


# =============================================================================
# 8. Write Silver Parquet
# =============================================================================

print(f"[WRITE] Overwriting Silver dataset: {TARGET_PATH}")

(
    silver_df
    .write
    .mode("overwrite")
    .parquet(TARGET_PATH)
)

print(
    "[SUCCESS] Provider Information Bronze -> Silver job completed."
)

job.commit()
