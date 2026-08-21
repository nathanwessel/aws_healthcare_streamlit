import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="CMS Nursing Home Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        .question-box {
            border-left: 4px solid #d0d7de;
            padding: 0.55rem 0.9rem;
            margin: 0.25rem 0 0.9rem 0;
            background: rgba(127, 127, 127, 0.04);
            border-radius: 0 0.35rem 0.35rem 0;
        }
        .question-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.65;
            margin-bottom: 0.15rem;
        }
        .question-text {
            font-size: 1rem;
            font-weight: 650;
            line-height: 1.35;
        }
        .scope-note {
            font-size: 0.82rem;
            opacity: 0.70;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(127, 127, 127, 0.22);
            padding: 0.8rem 0.9rem;
            border-radius: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Exact original questions retained verbatim for the dashboard.
QUESTION_2 = "Comparison of staffing levels vs. bed occupancy rates"
QUESTION_3 = "Facilities with the lowest staffing levels compared to patient load"
QUESTION_4 = "Correlation between nurse staffing levels and readmission rates"
QUESTION_5 = "Trend analysis of nurse attrition rates"


GOLD_PATHS = {
    "staffing_vs_occupancy": (
        "s3://nw-healthcare-project/gold/metrics/"
        "02_staffing_vs_occupancy/"
    ),
    "low_staffing": (
        "s3://nw-healthcare-project/gold/metrics/"
        "03_low_staffing_vs_patient_load/"
    ),
    "staffing_vs_readmissions": (
        "s3://nw-healthcare-project/gold/metrics/"
        "04_staffing_vs_readmissions/"
    ),
    "nurse_turnover": (
        "s3://nw-healthcare-project/gold/metrics/"
        "05_nurse_turnover_trends/"
    ),
}


EXPECTED_COLUMNS = {
    "staffing_vs_occupancy": {
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "MONTH",
        "NUMBER_OF_CERTIFIED_BEDS",
        "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
        "OCCUPANCY_RATE_PCT",
        "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
    },
    "low_staffing": {
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "MONTH",
        "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
        "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
        "NATIONAL_RANK",
        "STATE_RANK",
    },
    "staffing_vs_readmissions": {
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "PROVIDER_SNAPSHOT_DATE",
        "QRP_START_DATE",
        "QRP_END_DATE",
        "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
        "READMISSION_RATE_RSRR",
        "STATE_STAFFING_READMISSION_CORRELATION",
        "NATIONAL_STAFFING_READMISSION_CORRELATION",
    },
    "nurse_turnover": {
        "CCN",
        "PROVIDER_NAME",
        "STATE",
        "MONTH",
        "TOTAL_NURSING_STAFF_TURNOVER",
        "REGISTERED_NURSE_TURNOVER",
        "STATE_AVG_TOTAL_NURSING_STAFF_TURNOVER",
        "STATE_AVG_REGISTERED_NURSE_TURNOVER",
    },
}


def get_s3_storage_options() -> Optional[Dict]:
    """
    Local: if Streamlit secrets are absent, s3fs uses the normal AWS
    credential chain, including credentials configured by the AWS CLI.

    Streamlit Community Cloud: configure an [aws] secrets section with:
        aws_access_key_id
        aws_secret_access_key
        region_name
    An optional aws_session_token is also supported.
    """
    try:
        aws = st.secrets["aws"]
    except Exception:
        return None

    key = aws.get("aws_access_key_id")
    secret = aws.get("aws_secret_access_key")
    token = aws.get("aws_session_token")
    region = aws.get("region_name", "us-east-1")

    if not key or not secret:
        return None

    options = {
        "key": key,
        "secret": secret,
        "client_kwargs": {"region_name": region},
    }
    if token:
        options["token"] = token
    return options


@st.cache_data(ttl=900, show_spinner=False)
def read_gold_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        engine="pyarrow",
        storage_options=get_s3_storage_options(),
    )


def validate_schema(name: str, df: pd.DataFrame) -> None:
    missing = sorted(EXPECTED_COLUMNS[name] - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing expected Gold columns: {missing}")


@st.cache_data(ttl=900, show_spinner="Reading Gold Parquet datasets from S3...")
def load_all_gold_data():
    """
    Load only Questions 2-5. Question 1 is intentionally excluded because
    the current project has only one monthly Provider Information snapshot.
    """
    data = {
        name: read_gold_parquet(path)
        for name, path in GOLD_PATHS.items()
    }

    for name, df in data.items():
        validate_schema(name, df)

    for name in ["staffing_vs_occupancy", "low_staffing", "nurse_turnover"]:
        data[name]["MONTH"] = pd.to_datetime(data[name]["MONTH"], errors="coerce")

    for column in ["PROVIDER_SNAPSHOT_DATE", "QRP_START_DATE", "QRP_END_DATE"]:
        data["staffing_vs_readmissions"][column] = pd.to_datetime(
            data["staffing_vs_readmissions"][column],
            errors="coerce",
        )

    for df in data.values():
        df["CCN"] = df["CCN"].astype("string")
        df["STATE"] = df["STATE"].astype("string")
        df["PROVIDER_NAME"] = df["PROVIDER_NAME"].astype("string")

    return data


def question_box(question: str, context: Optional[str] = None) -> None:
    context_html = ""
    if context:
        context_html = (
            f"<div class='scope-note' style='margin-top:0.35rem;'>{context}</div>"
        )

    st.markdown(
        f"""
        <div class="question-box">
            <div class="question-label">Question this answers</div>
            <div class="question-text">{question}</div>
            {context_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(
    df: pd.DataFrame,
    selected_state: str,
    selected_ccn: Optional[str],
) -> pd.DataFrame:
    result = df.copy()
    if selected_state != "All States":
        result = result[result["STATE"] == selected_state]
    if selected_ccn is not None:
        result = result[result["CCN"] == selected_ccn]
    return result


def safe_mean(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def fmt_number(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def format_month(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return pd.Timestamp(value).strftime("%B %Y")


def correlation_strength(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "Not enough data to calculate a correlation."

    magnitude = abs(value)
    if magnitude < 0.20:
        strength = "very weak"
    elif magnitude < 0.40:
        strength = "weak"
    elif magnitude < 0.60:
        strength = "moderate"
    elif magnitude < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    if value > 0:
        direction = "positive"
    elif value < 0:
        direction = "negative"
    else:
        direction = "no"

    if direction == "no":
        return "There is essentially no linear relationship in this population."

    return (
        f"This is a {strength} {direction} linear relationship. "
        "Correlation does not establish causation."
    )


def add_linear_trendline(
    fig: go.Figure,
    x: pd.Series,
    y: pd.Series,
    name: str = "Linear trend",
) -> None:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2:
        return

    coefficients = np.polyfit(clean["x"].astype(float), clean["y"].astype(float), 1)
    x_line = np.linspace(clean["x"].min(), clean["x"].max(), 100)
    y_line = coefficients[0] * x_line + coefficients[1]

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=name,
        )
    )


def state_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Question 2 heatmap. Color uses standardized values because occupancy and
    staffing have different units; hover shows the actual values.
    """
    state_summary = (
        df.groupby("STATE", dropna=True)
        .agg(
            OCCUPANCY_RATE_PCT=("OCCUPANCY_RATE_PCT", "mean"),
            STAFFING_HOURS=("TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY", "mean"),
        )
        .dropna(how="all")
        .sort_index()
    )

    if state_summary.empty:
        return go.Figure()

    actual = state_summary[["OCCUPANCY_RATE_PCT", "STAFFING_HOURS"]].astype(float)
    standardized = actual.copy()

    for column in standardized.columns:
        std = standardized[column].std(ddof=0)
        mean = standardized[column].mean()
        if pd.isna(std) or math.isclose(std, 0.0):
            standardized[column] = 0.0
        else:
            standardized[column] = (standardized[column] - mean) / std

    x_labels = [
        "Average occupancy rate (%)",
        "Avg nursing staff hours / resident / day",
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=standardized.values,
            x=x_labels,
            y=standardized.index.tolist(),
            customdata=actual.values,
            colorbar=dict(title="Standardized<br>value"),
            hovertemplate=(
                "State: %{y}<br>"
                "%{x}<br>"
                "Actual value: %{customdata:.2f}<br>"
                "Standardized value: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(420, len(state_summary) * 18),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=None,
        yaxis_title=None,
    )
    return fig


# =============================================================================
# Load data
# =============================================================================

try:
    data = load_all_gold_data()
except Exception as exc:
    st.error("The dashboard could not read the Gold Parquet datasets from S3.")
    st.code(str(exc))
    st.info(
        "Locally, make sure your normal AWS CLI credentials can read "
        "s3://nw-healthcare-project/gold/metrics/. On Streamlit Community "
        "Cloud, configure the [aws] secrets used by this app."
    )
    st.stop()

staffing_occ_df = data["staffing_vs_occupancy"]
low_staffing_df = data["low_staffing"]
readmission_df = data["staffing_vs_readmissions"]
turnover_df = data["nurse_turnover"]


# =============================================================================
# Sidebar filters
# =============================================================================

st.sidebar.header("Filters")

states = sorted(staffing_occ_df["STATE"].dropna().astype(str).unique().tolist())
selected_state = st.sidebar.selectbox("State", ["All States"] + states)

facility_lookup = staffing_occ_df[["CCN", "PROVIDER_NAME", "STATE"]].drop_duplicates().copy()
if selected_state != "All States":
    facility_lookup = facility_lookup[facility_lookup["STATE"] == selected_state]

facility_lookup["DISPLAY"] = (
    facility_lookup["PROVIDER_NAME"].fillna("Unknown facility")
    + " ("
    + facility_lookup["CCN"].fillna("Unknown CCN")
    + ")"
)
facility_lookup = facility_lookup.sort_values(["PROVIDER_NAME", "CCN"])

selected_facility_label = st.sidebar.selectbox(
    "Facility",
    ["All Facilities"] + facility_lookup["DISPLAY"].tolist(),
)

selected_ccn = None
if selected_facility_label != "All Facilities":
    selected_ccn = facility_lookup.loc[
        facility_lookup["DISPLAY"] == selected_facility_label,
        "CCN",
    ].iloc[0]

if st.sidebar.button("Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "State and facility selections filter the dashboard. Relationship charts "
    "are most informative with All Facilities selected."
)


# =============================================================================
# Filtered data and header metadata
# =============================================================================

staffing_occ_filtered = apply_filters(staffing_occ_df, selected_state, selected_ccn)
low_staffing_filtered = apply_filters(low_staffing_df, selected_state, selected_ccn)
readmission_filtered = apply_filters(readmission_df, selected_state, selected_ccn)
turnover_filtered = apply_filters(turnover_df, selected_state, selected_ccn)

latest_snapshot = staffing_occ_df["MONTH"].max()
scope_text = selected_state if selected_state != "All States" else "All States"
if selected_facility_label != "All Facilities":
    scope_text += f" · {selected_facility_label}"


st.title("CMS Nursing Home Analytics Dashboard")
st.caption(
    f"Data snapshot: {format_month(latest_snapshot)} · Current scope: {scope_text}"
)
st.markdown(
    """
    This dashboard uses the Gold-layer metrics produced by the AWS Glue
    pipeline. It intentionally excludes the original occupancy-trend question
    because only one monthly Provider Information snapshot is currently
    available.
    """
)


tab_overview, tab_q2, tab_q3, tab_q4, tab_q5 = st.tabs(
    [
        "Overview",
        "Staffing vs. Occupancy",
        "Lowest Staffing",
        "Readmissions",
        "Nurse Turnover",
    ]
)


# =============================================================================
# OVERVIEW
# =============================================================================

with tab_overview:
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        question_box(QUESTION_2)
        st.metric(
            "Average Facility Occupancy Rate",
            fmt_number(safe_mean(staffing_occ_filtered["OCCUPANCY_RATE_PCT"]), 1, "%"),
        )

    with col2:
        question_box(QUESTION_2)
        st.metric(
            "Avg Nursing Staff Hours / Resident / Day",
            fmt_number(
                safe_mean(
                    staffing_occ_filtered[
                        "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"
                    ]
                ),
                2,
            ),
        )

    with col3:
        question_box(QUESTION_3)
        staffing_values = pd.to_numeric(
            low_staffing_filtered["TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"],
            errors="coerce",
        )
        lowest_staffing = float(staffing_values.min()) if staffing_values.notna().any() else None
        st.metric(
            "Lowest Staffing Level in Current Scope",
            fmt_number(lowest_staffing, 2),
        )

    with col4:
        question_box(QUESTION_5)
        st.metric(
            "Average Total Nursing Staff Turnover",
            fmt_number(safe_mean(turnover_filtered["TOTAL_NURSING_STAFF_TURNOVER"]), 1),
        )

    st.divider()

    question_box(
        QUESTION_2,
        "Heatmap colors are standardized within each metric because occupancy "
        "and staffing use different units. Hover to see the actual values.",
    )

    if staffing_occ_filtered.empty:
        st.warning("No staffing/occupancy data is available for this filter.")
    else:
        st.plotly_chart(state_heatmap(staffing_occ_filtered), use_container_width=True)


# =============================================================================
# QUESTION 2
# =============================================================================

with tab_q2:
    st.subheader("Staffing vs. Occupancy")

    question_box(
        QUESTION_2,
        "Each point is a facility. Staffing is measured as total nursing staff "
        "hours per resident per day, while occupancy is average residents "
        "divided by certified beds.",
    )

    q2_plot = (
        staffing_occ_filtered[
            [
                "CCN",
                "PROVIDER_NAME",
                "STATE",
                "OCCUPANCY_RATE_PCT",
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
                "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
                "NUMBER_OF_CERTIFIED_BEDS",
            ]
        ]
        .dropna(
            subset=[
                "OCCUPANCY_RATE_PCT",
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
            ]
        )
        .copy()
    )

    if q2_plot.empty:
        st.warning("No staffing and occupancy rows are available for this filter.")
    else:
        m1, m2 = st.columns(2)

        with m1:
            question_box(QUESTION_2)
            st.metric(
                "Average Facility Occupancy Rate",
                fmt_number(safe_mean(q2_plot["OCCUPANCY_RATE_PCT"]), 1, "%"),
            )

        with m2:
            question_box(QUESTION_2)
            st.metric(
                "Avg Nursing Staff Hours / Resident / Day",
                fmt_number(
                    safe_mean(q2_plot["TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"]),
                    2,
                ),
            )

        question_box(QUESTION_2)

        fig = px.scatter(
            q2_plot,
            x="OCCUPANCY_RATE_PCT",
            y="TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
            hover_name="PROVIDER_NAME",
            hover_data={
                "CCN": True,
                "STATE": True,
                "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY": ":.1f",
                "NUMBER_OF_CERTIFIED_BEDS": ":.0f",
                "OCCUPANCY_RATE_PCT": ":.1f",
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY": ":.2f",
            },
            labels={
                "OCCUPANCY_RATE_PCT": "Facility Occupancy Rate (%)",
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY": (
                    "Nursing Staff Hours per Resident per Day"
                ),
            },
            title="Facility Staffing vs. Occupancy",
        )

        add_linear_trendline(
            fig,
            q2_plot["OCCUPANCY_RATE_PCT"],
            q2_plot["TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"],
        )
        fig.update_layout(template="plotly_white", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        if len(q2_plot) < 2:
            st.caption(
                "Select All Facilities to see the cross-facility relationship "
                "and linear trendline."
            )


# =============================================================================
# QUESTION 3
# =============================================================================

with tab_q3:
    st.subheader("Lowest Staffing Relative to Patient Load")

    question_box(
        QUESTION_3,
        "CMS already normalizes staffing to resident load using total nursing "
        "staff hours per resident per day. Lower values indicate less staffing "
        "relative to patient load.",
    )

    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        ranking_scope = st.radio(
            "Ranking scope",
            ["National", "Within selected state"],
            horizontal=True,
        )
    with control_col2:
        ranking_count = st.selectbox("Facilities to show", [10, 25, 50], index=0)

    ranking_source = low_staffing_df.copy()
    if selected_state != "All States":
        ranking_source = ranking_source[ranking_source["STATE"] == selected_state]
    if selected_ccn is not None:
        ranking_source = ranking_source[ranking_source["CCN"] == selected_ccn]

    if ranking_scope == "Within selected state":
        rank_column = "STATE_RANK"
        if selected_state == "All States":
            st.info("Choose a specific State in the sidebar to use State Rank.")
            ranking_source = ranking_source.iloc[0:0]
    else:
        rank_column = "NATIONAL_RANK"

    ranking_source = (
        ranking_source
        .dropna(
            subset=[
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
                rank_column,
            ]
        )
        .sort_values(rank_column)
        .head(ranking_count)
        .copy()
    )

    if ranking_source.empty:
        st.warning("No ranked staffing rows are available for this selection.")
    else:
        question_box(QUESTION_3)
        bar = px.bar(
            ranking_source.sort_values(
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
                ascending=False,
            ),
            x="TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
            y="PROVIDER_NAME",
            orientation="h",
            hover_data={
                "CCN": True,
                "STATE": True,
                "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY": ":.1f",
                rank_column: True,
            },
            labels={
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY": (
                    "Nursing Staff Hours per Resident per Day"
                ),
                "PROVIDER_NAME": "Facility",
            },
            title="Facilities with the Lowest Staffing Levels Relative to Resident Load",
        )
        bar.update_layout(template="plotly_white", yaxis_title=None)
        st.plotly_chart(bar, use_container_width=True)

        question_box(QUESTION_3)
        table_columns = [
            rank_column,
            "PROVIDER_NAME",
            "STATE",
            "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY",
            "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
        ]
        st.dataframe(
            ranking_source[table_columns].rename(
                columns={
                    rank_column: "RANK",
                    "PROVIDER_NAME": "FACILITY",
                    "AVERAGE_NUMBER_OF_RESIDENTS_PER_DAY": "AVG_RESIDENTS_PER_DAY",
                    "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY": (
                        "STAFFING_HOURS_PER_RESIDENT_DAY"
                    ),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# QUESTION 4
# =============================================================================

with tab_q4:
    st.subheader("Staffing vs. Readmissions")

    question_box(
        QUESTION_4,
        "Readmission rate uses the Gold job's risk-standardized potentially "
        "preventable readmission measure. Correlation is a population-level "
        "relationship, not evidence that staffing causes readmission outcomes.",
    )

    q4_plot = (
        readmission_filtered[
            [
                "CCN",
                "PROVIDER_NAME",
                "STATE",
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
                "READMISSION_RATE_RSRR",
                "STATE_STAFFING_READMISSION_CORRELATION",
                "NATIONAL_STAFFING_READMISSION_CORRELATION",
            ]
        ]
        .dropna(
            subset=[
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
                "READMISSION_RATE_RSRR",
            ]
        )
        .copy()
    )

    if q4_plot.empty:
        st.warning("No matched staffing/readmission rows are available for this filter.")
    else:
        correlation_value = None
        correlation_label = "Correlation"

        if selected_ccn is None:
            if selected_state == "All States":
                values = pd.to_numeric(
                    q4_plot["NATIONAL_STAFFING_READMISSION_CORRELATION"],
                    errors="coerce",
                ).dropna()
                if not values.empty:
                    correlation_value = float(values.iloc[0])
                correlation_label = "National Pearson Correlation"
            else:
                values = pd.to_numeric(
                    q4_plot["STATE_STAFFING_READMISSION_CORRELATION"],
                    errors="coerce",
                ).dropna()
                if not values.empty:
                    correlation_value = float(values.iloc[0])
                correlation_label = f"{selected_state} Pearson Correlation"

        metric_col, explanation_col = st.columns([1, 2])

        with metric_col:
            question_box(QUESTION_4)
            st.metric(correlation_label, fmt_number(correlation_value, 3))

        with explanation_col:
            question_box(QUESTION_4)
            if selected_ccn is not None:
                st.info(
                    "A Pearson correlation cannot be calculated for one "
                    "facility. Select All Facilities to view the national or "
                    "selected-state correlation."
                )
            else:
                st.write(correlation_strength(correlation_value))

        question_box(QUESTION_4)
        fig = px.scatter(
            q4_plot,
            x="TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY",
            y="READMISSION_RATE_RSRR",
            hover_name="PROVIDER_NAME",
            hover_data={
                "CCN": True,
                "STATE": True,
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY": ":.2f",
                "READMISSION_RATE_RSRR": ":.2f",
            },
            labels={
                "TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY": (
                    "Nursing Staff Hours per Resident per Day"
                ),
                "READMISSION_RATE_RSRR": (
                    "Risk-Standardized Potentially Preventable Readmission Rate"
                ),
            },
            title="Nurse Staffing vs. Readmission Rate",
        )
        add_linear_trendline(
            fig,
            q4_plot["TOTAL_NURSE_STAFFING_HOURS_PER_RESIDENT_DAY"],
            q4_plot["READMISSION_RATE_RSRR"],
        )
        fig.update_layout(template="plotly_white", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# QUESTION 5
# =============================================================================

with tab_q5:
    st.subheader("Nurse Turnover")

    question_box(
        QUESTION_5,
        "The CMS source provides turnover measures. With only one monthly "
        "snapshot, a true time trend cannot yet be calculated. This section "
        "automatically switches to a trend chart when multiple months exist.",
    )

    q5_plot = turnover_filtered.copy()

    if q5_plot.empty:
        st.warning("No turnover data is available for this filter.")
    else:
        q5_plot["MONTH"] = pd.to_datetime(q5_plot["MONTH"], errors="coerce")

        kpi1, kpi2 = st.columns(2)
        with kpi1:
            question_box(QUESTION_5)
            st.metric(
                "Average Total Nursing Staff Turnover",
                fmt_number(safe_mean(q5_plot["TOTAL_NURSING_STAFF_TURNOVER"]), 1),
            )
        with kpi2:
            question_box(QUESTION_5)
            st.metric(
                "Average Registered Nurse Turnover",
                fmt_number(safe_mean(q5_plot["REGISTERED_NURSE_TURNOVER"]), 1),
            )

        month_count = q5_plot["MONTH"].dropna().nunique()

        if month_count >= 2:
            question_box(QUESTION_5)
            monthly_turnover = (
                q5_plot.groupby("MONTH", as_index=False)
                .agg(
                    TOTAL_NURSING_STAFF_TURNOVER=(
                        "TOTAL_NURSING_STAFF_TURNOVER",
                        "mean",
                    ),
                    REGISTERED_NURSE_TURNOVER=(
                        "REGISTERED_NURSE_TURNOVER",
                        "mean",
                    ),
                )
                .sort_values("MONTH")
            )
            long_turnover = monthly_turnover.melt(
                id_vars="MONTH",
                value_vars=[
                    "TOTAL_NURSING_STAFF_TURNOVER",
                    "REGISTERED_NURSE_TURNOVER",
                ],
                var_name="TURNOVER_TYPE",
                value_name="TURNOVER",
            )
            line = px.line(
                long_turnover,
                x="MONTH",
                y="TURNOVER",
                color="TURNOVER_TYPE",
                markers=True,
                labels={
                    "MONTH": "Month",
                    "TURNOVER": "Turnover",
                    "TURNOVER_TYPE": "Measure",
                },
                title="Nurse Turnover Trend Over Time",
            )
            line.update_layout(template="plotly_white", legend_title_text="")
            st.plotly_chart(line, use_container_width=True)

        else:
            st.info(
                f"Only {format_month(q5_plot['MONTH'].max())} is currently "
                "available. The original trend question cannot yet be answered "
                "over time. The comparison below shows the available turnover "
                "snapshot without presenting it as a time trend."
            )

            question_box(
                QUESTION_5,
                "Current single-snapshot comparison; this is not yet a time-series trend.",
            )

            if selected_ccn is None and selected_state == "All States":
                # Avoid a misleading "top 15 facilities" view where many
                # facilities can tie at the CMS 100% turnover ceiling.
                # State averages provide a more informative snapshot.
                state_turnover = (
                    q5_plot.groupby("STATE", as_index=False)
                    .agg(
                        TOTAL_NURSING_STAFF_TURNOVER=(
                            "TOTAL_NURSING_STAFF_TURNOVER",
                            "mean",
                        ),
                        REGISTERED_NURSE_TURNOVER=(
                            "REGISTERED_NURSE_TURNOVER",
                            "mean",
                        ),
                    )
                    .dropna(subset=["TOTAL_NURSING_STAFF_TURNOVER"])
                    .sort_values(
                        "TOTAL_NURSING_STAFF_TURNOVER",
                        ascending=False,
                    )
                    .head(15)
                )

                if not state_turnover.empty:
                    long_turnover = state_turnover.melt(
                        id_vars=["STATE"],
                        value_vars=[
                            "TOTAL_NURSING_STAFF_TURNOVER",
                            "REGISTERED_NURSE_TURNOVER",
                        ],
                        var_name="TURNOVER_TYPE",
                        value_name="TURNOVER_RATE_PCT",
                    )

                    long_turnover["TURNOVER_TYPE"] = (
                        long_turnover["TURNOVER_TYPE"].replace(
                            {
                                "TOTAL_NURSING_STAFF_TURNOVER":
                                    "Total Nursing Staff",
                                "REGISTERED_NURSE_TURNOVER":
                                    "Registered Nurses",
                            }
                        )
                    )

                    bar = px.bar(
                        long_turnover,
                        x="TURNOVER_RATE_PCT",
                        y="STATE",
                        color="TURNOVER_TYPE",
                        barmode="group",
                        orientation="h",
                        labels={
                            "TURNOVER_RATE_PCT":
                                "Average Annual Turnover Rate (%)",
                            "STATE": "State",
                            "TURNOVER_TYPE": "Measure",
                        },
                        title=(
                            "October 2024 Snapshot — States with the Highest "
                            "Average Annual Nursing Staff Turnover"
                        ),
                    )

                    state_order = state_turnover["STATE"].tolist()
                    bar.update_yaxes(
                        categoryorder="array",
                        categoryarray=state_order[::-1],
                    )

                    bar.update_layout(
                        template="plotly_white",
                        legend_title_text="",
                        yaxis_title=None,
                        xaxis_range=[0, 100],
                    )

                    st.plotly_chart(bar, use_container_width=True)

                    st.caption(
                        "These are state averages of the annual turnover "
                        "measures reported in the October 2024 CMS snapshot. "
                        "This is a cross-sectional comparison, not a time trend."
                    )

            elif selected_ccn is None:
                # When a state is selected, show the distribution across
                # facilities rather than another ranking dominated by ties.
                distribution = (
                    q5_plot[
                        [
                            "TOTAL_NURSING_STAFF_TURNOVER",
                            "REGISTERED_NURSE_TURNOVER",
                        ]
                    ]
                    .melt(
                        value_vars=[
                            "TOTAL_NURSING_STAFF_TURNOVER",
                            "REGISTERED_NURSE_TURNOVER",
                        ],
                        var_name="TURNOVER_TYPE",
                        value_name="TURNOVER_RATE_PCT",
                    )
                    .dropna(subset=["TURNOVER_RATE_PCT"])
                )

                distribution["TURNOVER_TYPE"] = (
                    distribution["TURNOVER_TYPE"].replace(
                        {
                            "TOTAL_NURSING_STAFF_TURNOVER":
                                "Total Nursing Staff",
                            "REGISTERED_NURSE_TURNOVER":
                                "Registered Nurses",
                        }
                    )
                )

                if not distribution.empty:
                    histogram = px.histogram(
                        distribution,
                        x="TURNOVER_RATE_PCT",
                        color="TURNOVER_TYPE",
                        barmode="overlay",
                        opacity=0.70,
                        nbins=20,
                        labels={
                            "TURNOVER_RATE_PCT":
                                "Annual Turnover Rate (%)",
                            "TURNOVER_TYPE": "Measure",
                        },
                        title=(
                            f"October 2024 Snapshot — Distribution of Annual "
                            f"Nursing Staff Turnover in {selected_state}"
                        ),
                    )

                    histogram.update_layout(
                        template="plotly_white",
                        legend_title_text="",
                        yaxis_title="Number of Facilities",
                    )
                    histogram.update_xaxes(range=[0, 100])

                    st.plotly_chart(histogram, use_container_width=True)

                    st.caption(
                        "This shows the distribution of annual turnover rates "
                        "across facilities in the selected state. It is not a "
                        "time-series trend."
                    )

            else:
                selected_row = q5_plot.iloc[0]

                detail = pd.DataFrame(
                    {
                        "Measure": [
                            "Total Nursing Staff",
                            "Registered Nurses",
                        ],
                        "Annual Turnover Rate (%)": [
                            selected_row["TOTAL_NURSING_STAFF_TURNOVER"],
                            selected_row["REGISTERED_NURSE_TURNOVER"],
                        ],
                    }
                )

                bar = px.bar(
                    detail,
                    x="Measure",
                    y="Annual Turnover Rate (%)",
                    title=(
                        "October 2024 Annual Turnover Snapshot — "
                        f"{selected_row['PROVIDER_NAME']}"
                    ),
                )

                bar.update_layout(
                    template="plotly_white",
                    xaxis_title=None,
                    yaxis_range=[0, 100],
                )

                st.plotly_chart(bar, use_container_width=True)


st.divider()

with st.expander("Data & methodology"):
    st.markdown(
        f"""
        **Questions included**

        - **{QUESTION_2}**
        - **{QUESTION_3}**
        - **{QUESTION_4}**
        - **{QUESTION_5}**

        **Question intentionally excluded**

        The original occupancy-trend question is not included because the
        project currently has only one monthly Provider Information snapshot.

        **Staffing measure**

        "Nursing Staff Hours per Resident per Day" is the CMS total nursing
        staffing measure used by the Gold pipeline. It represents total nursing
        staff hours normalized to resident load. It should not be described as
        a literal "1 nurse : X patients" ratio.

        **Occupancy**

        The Gold pipeline calculates facility occupancy as average residents
        per day divided by certified beds.

        **Readmissions**

        The readmission analysis uses the risk-standardized potentially
        preventable readmission measure selected in the Gold Glue job.

        **Turnover**

        The dashboard shows the CMS annual Total Nursing Staff Turnover and
        Registered Nurse Turnover rates. Until additional monthly snapshots
        are added, the turnover section uses cross-sectional state/facility
        comparisons rather than presenting the October 2024 release as a
        time trend.

        **Data source**

        The dashboard reads the four relevant Gold Parquet prefixes directly
        from Amazon S3. Data loading is cached for 15 minutes unless "Refresh
        data" is clicked.
        """
    )
