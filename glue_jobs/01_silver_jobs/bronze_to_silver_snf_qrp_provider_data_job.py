"""
AWS Glue PySpark job
Bronze -> Silver: CMS SNF QRP Provider Data
===========================================

This job is intentionally specific to the CMS:
    Skilled Nursing Facility Quality Reporting Program - Provider Data

It will NOT process NH_ProviderInfo files.

Current source example
----------------------
s3://nw-healthcare-project/bronze/Nursing_Homes_data/
Skilled_Nursing_Facility_Quality_Reporting_Program_Provider_Data_Oct2024.csv

Default Silver destination
--------------------------
s3://nw-healthcare-project/silver/snf_qrp_provider_data/

What the job does
-----------------
1. Finds all monthly SNF QRP Provider Data CSVs directly under the Bronze prefix.
2. Reads each snapshot independently as raw STRING data.
3. Converts headers to UPPER_SNAKE_CASE.
4. Maps "CMS Certification Number (CCN)" to simply "CCN".
5. Verifies that the QRP-specific columns exist:
       MEASURE_CODE
       SCORE
       START_DATE
       END_DATE
   This safety check prevents accidentally processing Provider Information data.
6. Keeps every source column.
7. Applies the October 2024 CMS data-dictionary types.
8. Derives SNAPSHOT_DATE from the filename:
       ..._Oct2024.csv -> 2024-10-01
9. Adds SOURCE_FILE_NAME and SILVER_PROCESSED_AT.
10. Writes the complete Silver dataset to Parquet using overwrite mode.

Important type choices
----------------------
* CCN remains STRING.
* SCORE remains STRING because CMS defines Score as Text.
* FOOTNOTE remains STRING. Although the CMS dictionary labels Footnote as
  Numeric, it also says multiple codes can be comma-separated (e.g. "9,13"),
  so STRING preserves the actual data correctly.
* START_DATE and END_DATE become DATE.
* ZIP_CODE and CMS_REGION are dictionary Numeric fields and become DOUBLE.
* Unknown future columns are preserved as STRING rather than dropped.

No deduplication, partitioning, coalesce(1), or Glue Catalog table creation
is performed.
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
# 1. Initialize AWS Glue / Spark
# =============================================================================

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(args["JOB_NAME"], args)


# =============================================================================
# 2. Defaults and optional Glue job parameters
# =============================================================================

DEFAULT_SOURCE_PREFIX = (
    "s3://nw-healthcare-project/bronze/Nursing_Homes_data/"
)

DEFAULT_TARGET_PATH = (
    "s3://nw-healthcare-project/silver/snf_qrp_provider_data/"
)


def get_optional_arg(name: str, default: str) -> str:
    """
    Read an optional Glue argument.

    Example:
        --SOURCE_PREFIX s3://bucket/bronze/Nursing_Homes_data/
        --TARGET_PATH   s3://bucket/silver/snf_qrp_provider_data/
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


# Only these monthly files are allowed into this Silver dataset.
QRP_FILE_PATTERN = re.compile(
    r"^Skilled_Nursing_Facility_Quality_Reporting_Program_"
    r"Provider_Data_([A-Za-z]+)([0-9]{4})\.csv$",
    re.IGNORECASE,
)


# =============================================================================
# 3. October 2024 CMS SNF QRP Provider Data schema
# =============================================================================

# CMS Text fields.
# ADDRESS_LINE_2 is documented for Swing Bed data, but keeping it in the map
# makes the transformation tolerant if CMS supplies it in a future file.
TEXT_COLUMNS = {
    "CCN",
    "PROVIDER_NAME",
    "ADDRESS_LINE_1",
    "ADDRESS_LINE_2",
    "CITY_TOWN",
    "STATE",
    "COUNTY_PARISH",
    "TELEPHONE_NUMBER",
    "MEASURE_CODE",
    "SCORE",
    "FOOTNOTE",
    "MEASURE_DATE_RANGE",
    "LOCATION1",
}

# CMS Date fields.
DATE_COLUMNS = {
    "START_DATE",
    "END_DATE",
}

# CMS Numeric fields used by this file.
NUMERIC_COLUMNS = {
    "ZIP_CODE",
    "CMS_REGION",
}

KNOWN_DICTIONARY_COLUMNS = (
    TEXT_COLUMNS
    | DATE_COLUMNS
    | NUMERIC_COLUMNS
)

# These four columns prove we actually loaded the QRP Provider Data file.
QRP_REQUIRED_SOURCE_COLUMNS = {
    "MEASURE_CODE",
    "SCORE",
    "START_DATE",
    "END_DATE",
}

METADATA_COLUMNS = {
    "SOURCE_FILE_NAME",
    "SNAPSHOT_DATE",
    "SILVER_PROCESSED_AT",
}

TYPED_NULL_TOKENS = {
    "",
    "NA",
    "N/A",
    "DATA NOT AVAILABLE",
}


# =============================================================================
# 4. S3 helpers
# =============================================================================

def parse_s3_uri(uri: str):
    """Convert s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, received: {uri}")

    remainder = uri[5:]
    pieces = remainder.split("/", 1)

    bucket = pieces[0]
    key = pieces[1] if len(pieces) > 1 else ""

    return bucket, key


def list_qrp_source_files(source_prefix: str):
    """
    List only SNF QRP Provider Data monthly CSVs directly in SOURCE_PREFIX.

    This is the main protection against accidentally processing
    NH_ProviderInfo_Oct2024.csv.
    """
    bucket, prefix = parse_s3_uri(source_prefix)

    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative_name = key[len(prefix):]

            # Ignore anything in nested folders.
            if "/" in relative_name:
                continue

            if QRP_FILE_PATTERN.match(relative_name):
                files.append(f"s3://{bucket}/{key}")

    files.sort()

    if not files:
        raise FileNotFoundError(
            "No SNF QRP Provider Data files were found under "
            f"{source_prefix}. Expected names like "
            "Skilled_Nursing_Facility_Quality_Reporting_Program_"
            "Provider_Data_Oct2024.csv"
        )

    return files


# =============================================================================
# 5. Column and filename helpers
# =============================================================================

def standardize_column_name(name: str) -> str:
    """
    Convert a CMS column header to UPPER_SNAKE_CASE.

    Examples:
        "Provider Name" -> PROVIDER_NAME
        "City/Town" -> CITY_TOWN
        "Measure Code" -> MEASURE_CODE
        "CMS Certification Number (CCN)" -> CCN
    """
    cleaned = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        name.strip(),
    )
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_").upper()

    if cleaned == "CMS_CERTIFICATION_NUMBER_CCN":
        return "CCN"

    return cleaned


def standardize_columns(df: DataFrame) -> DataFrame:
    """Rename every source column and reject normalization collisions."""
    new_names = [
        standardize_column_name(column)
        for column in df.columns
    ]

    if len(new_names) != len(set(new_names)):
        raise ValueError(
            "At least two raw columns normalize to the same "
            f"UPPER_SNAKE_CASE name. Raw columns: {df.columns}"
        )

    for old_name, new_name in zip(df.columns, new_names):
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)

    return df


def validate_qrp_columns(df: DataFrame, source_file_name: str) -> None:
    """
    Fail immediately if the loaded file is not really QRP Provider Data.

    This check would have caught the earlier problem before bad data reached
    the QRP Silver path.
    """
    missing = sorted(
        QRP_REQUIRED_SOURCE_COLUMNS - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{source_file_name} does not look like SNF QRP Provider Data. "
            f"Missing QRP columns: {missing}. "
            f"Columns actually found: {df.columns}"
        )


def snapshot_date_from_filename(file_name: str):
    """
    Example:
        ..._Provider_Data_Oct2024.csv -> 2024-10-01
    """
    match = QRP_FILE_PATTERN.match(file_name)

    if not match:
        raise ValueError(
            f"Filename does not match the QRP monthly pattern: {file_name}"
        )

    month_text = match.group(1)
    year_text = match.group(2)
    month_year = f"{month_text}{year_text}"

    for fmt in ("%b%Y", "%B%Y"):
        try:
            parsed = datetime.strptime(month_year, fmt)
            return parsed.date().replace(day=1)
        except ValueError:
            pass

    raise ValueError(
        f"Could not parse snapshot month from filename: {file_name}"
    )


# =============================================================================
# 6. Data cleaning/type helpers
# =============================================================================

def trim_text(column_name: str):
    """For text fields, turn only blank strings into NULL."""
    value = F.trim(F.col(column_name))

    return F.when(
        value == "",
        F.lit(None).cast("string"),
    ).otherwise(value)


def typed_null(column_name: str):
    """Identify configured NULL-like values for Date/Numeric fields."""
    value = F.upper(
        F.trim(F.col(column_name).cast("string"))
    )

    return value.isin(*TYPED_NULL_TOKENS)


def clean_numeric(column_name: str):
    """
    Convert a CMS Numeric field to DOUBLE.

    Remove common display formatting first.
    """
    raw = F.col(column_name).cast("string")

    cleaned = F.trim(raw)
    cleaned = F.regexp_replace(cleaned, ",", "")
    cleaned = F.regexp_replace(cleaned, r"\$", "")
    cleaned = F.regexp_replace(cleaned, "%", "")

    return F.when(
        typed_null(column_name),
        F.lit(None).cast("double"),
    ).otherwise(
        cleaned.cast("double")
    )


def clean_date(column_name: str):
    """Convert common CMS date representations into a Spark DATE."""
    value = F.trim(F.col(column_name).cast("string"))

    parsed = F.coalesce(
        F.to_date(value, "MM/dd/yyyy"),
        F.to_date(value, "M/d/yyyy"),
        F.to_date(value, "yyyy-MM-dd"),
        F.to_date(value, "MM-dd-yyyy"),
    )

    return F.when(
        typed_null(column_name),
        F.lit(None).cast("date"),
    ).otherwise(parsed)


def apply_dictionary_types(df: DataFrame) -> DataFrame:
    """
    Apply known CMS types and preserve unknown future source columns as STRING.
    """
    unknown_columns = sorted(
        set(df.columns)
        - KNOWN_DICTIONARY_COLUMNS
        - METADATA_COLUMNS
    )

    if unknown_columns:
        print(
            "[WARNING] Future/unmapped source columns will be kept as STRING:"
        )
        for column in unknown_columns:
            print(f"  - {column}")

    for column in df.columns:
        if column in TEXT_COLUMNS:
            df = df.withColumn(
                column,
                trim_text(column),
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


def union_allow_missing_columns(dataframes):
    """
    Union monthly snapshots while tolerating future schema drift.

    Missing future/past columns are inserted as STRING NULLs.
    """
    if not dataframes:
        raise ValueError("No monthly QRP DataFrames were supplied.")

    ordered_columns = []

    for df in dataframes:
        for column in df.columns:
            if column not in ordered_columns:
                ordered_columns.append(column)

    aligned = []

    for df in dataframes:
        for column in ordered_columns:
            if column not in df.columns:
                df = df.withColumn(
                    column,
                    F.lit(None).cast("string"),
                )

        aligned.append(
            df.select(*ordered_columns)
        )

    result = aligned[0]

    for df in aligned[1:]:
        result = result.unionByName(df)

    return result


# =============================================================================
# 7. Discover and read the Bronze QRP files
# =============================================================================

source_files = list_qrp_source_files(SOURCE_PREFIX)

print("=" * 80)
print("[INFO] QRP Bronze -> Silver job configuration")
print(f"[INFO] SOURCE_PREFIX = {SOURCE_PREFIX}")
print(f"[INFO] TARGET_PATH   = {TARGET_PATH}")
print(f"[INFO] Matching QRP files found: {len(source_files)}")
for source_file in source_files:
    print(f"[INFO]   {source_file}")
print("=" * 80)

monthly_dataframes = []

for source_uri in source_files:
    source_file_name = source_uri.rsplit("/", 1)[-1]
    snapshot_date = snapshot_date_from_filename(
        source_file_name
    )

    print(
        f"[READ] {source_uri} | SNAPSHOT_DATE={snapshot_date}"
    )

    # Read raw Bronze as STRING. Silver owns the explicit type conversion.
    current_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("quote", '"')
        .option("escape", '"')
        .csv(source_uri)
    )

    current_df = standardize_columns(current_df)

    # Critical safety check.
    validate_qrp_columns(
        current_df,
        source_file_name,
    )

    reserved_collision = (
        METADATA_COLUMNS.intersection(current_df.columns)
    )

    if reserved_collision:
        raise ValueError(
            "Raw source unexpectedly contains reserved Silver metadata "
            f"columns: {sorted(reserved_collision)}"
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
# 8. Build the final Silver dataset
# =============================================================================

silver_df = union_allow_missing_columns(
    monthly_dataframes
)

silver_df = silver_df.withColumn(
    "SILVER_PROCESSED_AT",
    F.current_timestamp(),
)

silver_df = apply_dictionary_types(
    silver_df
)

# Safety check again after all transformations.
validate_qrp_columns(
    silver_df,
    "final Silver DataFrame",
)


# =============================================================================
# 9. Log the exact resulting schema
# =============================================================================

row_count = silver_df.count()

print("=" * 80)
print(f"[INFO] Final Silver row count: {row_count}")
print(f"[INFO] Final Silver column count: {len(silver_df.columns)}")
print("[INFO] Final Silver columns:")
for column in silver_df.columns:
    print(f"[INFO]   {column}")
print("[INFO] Final Silver schema:")
silver_df.printSchema()
print("=" * 80)


# =============================================================================
# 10. Overwrite the QRP Silver path with Parquet
# =============================================================================

print(f"[WRITE] Overwriting QRP Silver dataset: {TARGET_PATH}")

(
    silver_df
    .write
    .mode("overwrite")
    .parquet(TARGET_PATH)
)

print(
    "[SUCCESS] SNF QRP Provider Data Bronze -> Silver completed successfully."
)

job.commit()
