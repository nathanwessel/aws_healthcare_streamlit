"""
AWS Glue PySpark job: Gold healthcare metrics
==============================================

Purpose
-------
Read the two Silver-layer Parquet datasets and produce five Gold-layer
Parquet datasets that can be read directly from S3 by a Streamlit app.

Silver inputs:
1. Provider Information
   s3://nw-healthcare-project/silver/provider_info/

2. SNF QRP Provider Data
   s3://nw-healthcare-project/silver/snf_qrp_provider_data/

Gold outputs:
1. occupancy_trends/
2. staffing_vs_occupancy/
3. low_staffing_vs_patient_load/
4. staffing_vs_readmissions/
5. nurse_turnover_trends/

Important note about the current data
-------------------------------------
The current project has only one snapshot: October 2024.

Therefore:
- "occupancy trends" currently contains one month per facility.
- "nurse turnover trends" currently contains one month per facility.
- The code is written so that additional monthly Silver snapshots can be
  added later without changing this Gold job.

Expected Silver column naming convention
----------------------------------------
The future Bronze -> Silver Glue jobs should standardize columns to
UPPERCASE_SNAKE_CASE.

Provider Information Silver schema used by this job:
    CCN
    PROVIDER_NAME
    STATE
    NUMBER_OF_CERTIFIED_BEDS
    AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY
    REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY
    TOTAL_NURSING_STAFF_TURNOVER
    REGISTERED_NURSE_TURNOVER
    SNAPSHOT_DATE

SNF QRP Provider Data Silver schema used by this job:
    CCN
    PROVIDER_NAME
    STATE
    MEASURE_CODE
    SCORE
    START_DATE
    END_DATE
    SNAPSHOT_DATE

SNAPSHOT_DATE should be created by the Silver jobs from the source filename,
for example:
    NH_ProviderInfo_Oct2024.csv -> 2024-10-01
    Skilled_Nursing_Facility_Quality_Reporting_Program_Provider_Data_Oct2024.csv
        -> 2024-10-01

The job uses TOTAL nursing staff (nurse aide + LPN + RN), matching the agreed
definition for "nurse staffing" in this project.
"""

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


# =============================================================================
# 1. Glue / Spark setup
# =============================================================================

# JOB_NAME is the only required Glue argument.
required_args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(required_args["JOB_NAME"], required_args)


# =============================================================================
# 2. Config / Glue job arguments
# =============================================================================

# These defaults point to the S3 paths agreed on for this project.
# They can be overridden in Glue Job Parameters without editing this file.
DEFAULT_PROVIDER_INFO_PATH = (
    "s3://nw-healthcare-project/silver/provider_info/"
)

DEFAULT_SNF_QRP_PATH = (
    "s3://nw-healthcare-project/silver/snf_qrp_provider_data/"
)

DEFAULT_GOLD_BASE_PATH = (
    "s3://nw-healthcare-project/gold/metrics/"
)

DEFAULT_LOOKBACK_MONTHS = 12


def get_optional_arg(name: str, default: str) -> str:
    """
    Read an optional Glue job parameter.

    Example:
        --GOLD_BASE_PATH s3://another-bucket/gold/metrics/

    If the parameter is not passed, use the supplied default.
    """
    flag = f"--{name}"

    if flag not in sys.argv:
        return default

    position = sys.argv.index(flag)

    if position + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for Glue argument {flag}")

    return sys.argv[position + 1]


PROVIDER_INFO_PATH = get_optional_arg(
    "PROVIDER_INFO_PATH",
    DEFAULT_PROVIDER_INFO_PATH,
).rstrip("/") + "/"

SNF_QRP_PATH = get_optional_arg(
    "SNF_QRP_PATH",
    DEFAULT_SNF_QRP_PATH,
).rstrip("/") + "/"

GOLD_BASE_PATH = get_optional_arg(
    "GOLD_BASE_PATH",
    DEFAULT_GOLD_BASE_PATH,
).rstrip("/") + "/"

LOOKBACK_MONTHS = int(
    get_optional_arg(
        "LOOKBACK_MONTHS",
        str(DEFAULT_LOOKBACK_MONTHS),
    )
)


# =============================================================================
# 3. Expected Silver schemas
# =============================================================================

PROVIDER_REQUIRED_COLUMNS = [
    "CCN",
    "PROVIDER_NAME",
    "STATE",
    "NUMBER_OF_CERTIFIED_BEDS",
    "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
    "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
    "TOTAL_NURSING_STAFF_TURNOVER",
    "REGISTERED_NURSE_TURNOVER",
    "SNAPSHOT_DATE",
]

QRP_REQUIRED_COLUMNS = [
    "CCN",
    "PROVIDER_NAME",
    "STATE",
    "MEASURE_CODE",
    "SCORE",
    "START_DATE",
    "END_DATE",
    "SNAPSHOT_DATE",
]


def require_columns(df: DataFrame, expected_columns: list[str], dataset_name: str) -> None:
    """
    Fail early with a useful error if the future Silver jobs do not produce
    the schema expected by this Gold job.

    This is a schema contract check, not a broader data-quality framework.
    """
    missing = [column for column in expected_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required Silver columns: {missing}"
        )


# =============================================================================
# 4. Read Silver data
# =============================================================================

provider_df = spark.read.parquet(PROVIDER_INFO_PATH)
qrp_df = spark.read.parquet(SNF_QRP_PATH)

require_columns(
    provider_df,
    PROVIDER_REQUIRED_COLUMNS,
    "Provider Information",
)

require_columns(
    qrp_df,
    QRP_REQUIRED_COLUMNS,
    "SNF QRP Provider Data",
)


# Normalize data types used by the calculations.
# The Silver jobs should already do this, but these casts make the Gold job
# more defensive and make the calculation intent obvious.
provider_df = (
    provider_df
    .withColumn("CCN", F.col("CCN").cast("string"))
    .withColumn("SNAPSHOT_DATE", F.to_date("SNAPSHOT_DATE"))
    .withColumn(
        "NUMBER_OF_CERTIFIED_BEDS",
        F.col("NUMBER_OF_CERTIFIED_BEDS").cast("double"),
    )
    .withColumn(
        "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
        F.col("AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY").cast("double"),
    )
    .withColumn(
        "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY",
        F.col(
            "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY"
        ).cast("double"),
    )
    .withColumn(
        "TOTAL_NURSING_STAFF_TURNOVER",
        F.col("TOTAL_NURSING_STAFF_TURNOVER").cast("double"),
    )
    .withColumn(
        "REGISTERED_NURSE_TURNOVER",
        F.col("REGISTERED_NURSE_TURNOVER").cast("double"),
    )
)

qrp_df = (
    qrp_df
    .withColumn("CCN", F.col("CCN").cast("string"))
    .withColumn("SNAPSHOT_DATE", F.to_date("SNAPSHOT_DATE"))
    .withColumn("START_DATE", F.to_date("START_DATE"))
    .withColumn("END_DATE", F.to_date("END_DATE"))
)

# =============================================================================
# 5. Shared helper columns / datasets
# =============================================================================

# Find the latest Provider Information snapshot available.
# With the current data this will be 2024-10-01.
latest_snapshot_row = (
    provider_df
    .agg(F.max("SNAPSHOT_DATE").alias("LATEST_SNAPSHOT_DATE"))
    .first()
)

LATEST_SNAPSHOT_DATE = latest_snapshot_row["LATEST_SNAPSHOT_DATE"]

if LATEST_SNAPSHOT_DATE is None:
    raise ValueError("Provider Information contains no valid SNAPSHOT_DATE values.")

# Keep the most recent N months.
#
# Example with LOOKBACK_MONTHS = 12:
# latest snapshot = 2024-10-01
# earliest allowed month = 2023-11-01
#
# Today there is only one month, so only Oct 2024 will be returned.
provider_lookback_df = (
    provider_df
    .filter(
        F.col("SNAPSHOT_DATE")
        >= F.add_months(
            F.lit(LATEST_SNAPSHOT_DATE),
            -(LOOKBACK_MONTHS - 1),
        )
    )
    .withColumn(
        "MONTH",
        F.trunc("SNAPSHOT_DATE", "month"),
    )
    .cache()
)

# Occupancy:
#
#     average residents per day
#     -------------------------  x 100
#        certified beds
#
# Do not divide by zero.
occupancy_expression = (
    F.when(
        F.col("NUMBER_OF_CERTIFIED_BEDS") > 0,
        (
            F.col("AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY")
            / F.col("NUMBER_OF_CERTIFIED_BEDS")
        )
        * F.lit(100.0),
    )
    .otherwise(F.lit(None).cast("double"))
)


def write_gold(df: DataFrame, folder_name: str) -> None:
    """
    Overwrite one Gold metric dataset as Parquet.

    Streamlit can later read the entire S3 prefix as a Parquet dataset.
    """
    output_path = f"{GOLD_BASE_PATH}{folder_name}/"

    (
        df
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    print(f"[WRITE] {folder_name} -> {output_path}")


# =============================================================================
# METRIC 1
# Hospital / facility occupancy rate trends by month
# =============================================================================
#
# Current output grain:
#     one row per CCN per month
#
# Because only Oct 2024 exists today, each facility will currently have one row.
# Once future monthly snapshots are added to Silver, this becomes a true trend.

metric_1_occupancy_trends = (
    provider_lookback_df
    .select(
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "MONTH",
        "NUMBER_OF_CERTIFIED_BEDS",
        "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
        occupancy_expression.alias("OCCUPANCY_RATE_PCT"),
    )
    .orderBy("STATE", "PROVIDER_NAME", "MONTH")
)

write_gold(
    metric_1_occupancy_trends,
    "01_occupancy_trends",
)

# =============================================================================
# METRIC 2
# Compare staffing levels vs. bed occupancy rates
# =============================================================================
#
# Current output grain:
#     one row per CCN per month
#
# Streamlit use:
#     X axis -> OCCUPANCY_RATE_PCT
#     Y axis -> TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY
#
# No Pearson correlation is calculated for this metric, per the agreed design.

metric_2_staffing_vs_occupancy = (
    provider_lookback_df
    .select(
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "MONTH",
        "NUMBER_OF_CERTIFIED_BEDS",
        "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
        occupancy_expression.alias("OCCUPANCY_RATE_PCT"),
        F.col(
            "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY"
        ).alias(
            "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"
        ),
    )
    .orderBy("MONTH", "STATE", "PROVIDER_NAME")
)

write_gold(
    metric_2_staffing_vs_occupancy,
    "02_staffing_vs_occupancy",
)


# =============================================================================
# METRIC 3
# Facilities with the lowest staffing levels compared to patient load
# =============================================================================
#
# CMS already normalizes total nursing staff to patient load as:
#     nursing staff hours per resident per day
#
# Therefore, the lowest value is the lowest staffing level relative to resident
# load. We rank ALL facilities rather than keeping only the bottom 10.
#
# We produce:
#     - NATIONAL_RANK
#     - STATE_RANK
#
# This metric uses the latest snapshot only so the ranking is not duplicated
# across months.

latest_provider_df = provider_df.filter(
    F.col("SNAPSHOT_DATE") == F.lit(LATEST_SNAPSHOT_DATE)
)

# First create the Streamlit-friendly staffing column name.
#
# Important:
# The ranking windows below must reference the column names that actually exist
# at the point where the window is evaluated. The source CMS column is renamed
# here to TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY, so the windows use that
# alias rather than the original source column name.
metric_3_rank_base = (
    latest_provider_df
    .filter(
        F.col(
            "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY"
        ).isNotNull()
    )
    .select(
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        F.col("SNAPSHOT_DATE").alias("MONTH"),
        "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
        F.col(
            "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY"
        ).alias(
            "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"
        ),
    )
)

national_rank_window = Window.orderBy(
    F.col(
        "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"
    ).asc_nulls_last(),
    F.col("CCN").asc(),
)

state_rank_window = Window.partitionBy("STATE").orderBy(
    F.col(
        "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"
    ).asc_nulls_last(),
    F.col("CCN").asc(),
)

metric_3_low_staffing = (
    metric_3_rank_base
    .withColumn(
        "NATIONAL_RANK",
        F.row_number().over(national_rank_window),
    )
    .withColumn(
        "STATE_RANK",
        F.row_number().over(state_rank_window),
    )
    .orderBy("NATIONAL_RANK")
)

write_gold(
    metric_3_low_staffing,
    "03_low_staffing_vs_patient_load",
)


# =============================================================================
# METRIC 4
# Correlation between nurse staffing levels and readmission rates
# =============================================================================
#
# Readmission measure:
#     S_004_01_PPR_PD_RSRR
#
# This is the risk-standardized potentially preventable readmission rate.
#
# Output:
#     - facility-level staffing and readmission values for a Streamlit scatterplot
#     - state-level Pearson correlation
#     - national Pearson correlation
#
# The correlation values are repeated onto each facility row so Streamlit can
# read a single Parquet dataset for this metric.
#
# Temporal matching:
#     1. Keep the latest available QRP readmission measure per CCN.
#     2. Match it to the Provider Information snapshot closest to the QRP
#        reporting END_DATE.
#
# With only one Provider Information snapshot today, every match will naturally
# use Oct 2024.

READMISSION_MEASURE_CODE = "S_004_01_PPR_PD_RSRR"

qrp_readmission_df = (
    qrp_df
    .filter(F.col("MEASURE_CODE") == READMISSION_MEASURE_CODE)
    .withColumn(
        "READMISSION_RATE_RSRR",
        F.col("SCORE").cast("double"),
    )
    .filter(F.col("READMISSION_RATE_RSRR").isNotNull())
)

latest_qrp_window = Window.partitionBy("CCN").orderBy(
    F.col("END_DATE").desc_nulls_last(),
    F.col("SNAPSHOT_DATE").desc_nulls_last(),
)

latest_qrp_readmission_df = (
    qrp_readmission_df
    .withColumn(
        "_QRP_ROW_NUMBER",
        F.row_number().over(latest_qrp_window),
    )
    .filter(F.col("_QRP_ROW_NUMBER") == 1)
    .drop("_QRP_ROW_NUMBER")
    .select(
        "CCN",
        F.col("START_DATE").alias("QRP_START_DATE"),
        F.col("END_DATE").alias("QRP_END_DATE"),
        F.col("SNAPSHOT_DATE").alias("QRP_SNAPSHOT_DATE"),
        "READMISSION_RATE_RSRR",
    )
)

provider_for_readmission_df = (
    provider_df
    .select(
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        F.col("SNAPSHOT_DATE").alias("PROVIDER_SNAPSHOT_DATE"),
        F.col(
            "REPORTED_TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_PER_DAY"
        ).alias(
            "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"
        ),
    )
    .filter(
        F.col("TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY").isNotNull()
    )
)

# Build candidate provider/QRP matches by CCN.
readmission_candidates = (
    provider_for_readmission_df.alias("P")
    .join(
        latest_qrp_readmission_df.alias("Q"),
        on="CCN",
        how="inner",
    )
    .withColumn(
        "_DAYS_FROM_QRP_END",
        F.abs(
            F.datediff(
                F.col("PROVIDER_SNAPSHOT_DATE"),
                F.col("QRP_END_DATE"),
            )
        ),
    )
)

closest_snapshot_window = Window.partitionBy("CCN").orderBy(
    F.col("_DAYS_FROM_QRP_END").asc_nulls_last(),
    F.col("PROVIDER_SNAPSHOT_DATE").desc_nulls_last(),
)

readmission_detail_df = (
    readmission_candidates
    .withColumn(
        "_MATCH_ROW_NUMBER",
        F.row_number().over(closest_snapshot_window),
    )
    .filter(F.col("_MATCH_ROW_NUMBER") == 1)
    .drop("_MATCH_ROW_NUMBER", "_DAYS_FROM_QRP_END")
    .select(
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "PROVIDER_SNAPSHOT_DATE",
        "QRP_START_DATE",
        "QRP_END_DATE",
        "QRP_SNAPSHOT_DATE",
        "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
        "READMISSION_RATE_RSRR",
    )
)

# Pearson correlation by state.
state_correlation_df = (
    readmission_detail_df
    .groupBy("STATE")
    .agg(
        F.corr(
            "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
            "READMISSION_RATE_RSRR",
        ).alias("STATE_STAFFING_READMISSION_CORRELATION"),
        F.count(F.lit(1)).alias("STATE_OBSERVATION_COUNT"),
    )
)

# Pearson correlation across all facilities nationally.
national_correlation_df = (
    readmission_detail_df
    .agg(
        F.corr(
            "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
            "READMISSION_RATE_RSRR",
        ).alias("NATIONAL_STAFFING_READMISSION_CORRELATION"),
        F.count(F.lit(1)).alias("NATIONAL_OBSERVATION_COUNT"),
    )
)

metric_4_staffing_vs_readmissions = (
    readmission_detail_df
    .join(
        state_correlation_df,
        on="STATE",
        how="left",
    )
    .crossJoin(national_correlation_df)
    .orderBy("STATE", "PROVIDER_NAME")
)

write_gold(
    metric_4_staffing_vs_readmissions,
    "04_staffing_vs_readmissions",
)


# =============================================================================
# METRIC 5
# Trend analysis of nurse turnover / attrition measures
# =============================================================================
#
# The source fields are turnover measures:
#     TOTAL_NURSING_STAFF_TURNOVER
#     REGISTERED_NURSE_TURNOVER
#
# We retain the facility-level values and also calculate state averages for
# the same month so Streamlit can show either:
#     - an individual facility
#     - a state comparison
#
# Today there is only one month (Oct 2024), so this is currently a one-point
# trend. The same code will automatically become a multi-month trend as new
# Silver snapshots are added.

state_month_window = Window.partitionBy(
    "STATE",
    "MONTH",
)

metric_5_nurse_turnover = (
    provider_lookback_df
    .select(
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "MONTH",
        "TOTAL_NURSING_STAFF_TURNOVER",
        "REGISTERED_NURSE_TURNOVER",
    )
    .withColumn(
        "STATE_AVG_TOTAL_NURSING_STAFF_TURNOVER",
        F.avg("TOTAL_NURSING_STAFF_TURNOVER").over(state_month_window),
    )
    .withColumn(
        "STATE_AVG_REGISTERED_NURSE_TURNOVER",
        F.avg("REGISTERED_NURSE_TURNOVER").over(state_month_window),
    )
    .orderBy("STATE", "PROVIDER_NAME", "MONTH")
)

write_gold(
    metric_5_nurse_turnover,
    "05_nurse_turnover_trends",
)


# =============================================================================
# 6. Cleanup and Glue commit
# =============================================================================

provider_lookback_df.unpersist()

job.commit()

print("[SUCCESS] All five Gold metric datasets were written successfully.")
