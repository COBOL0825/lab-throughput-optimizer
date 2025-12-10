# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path("centrifuges.csv")

st.set_page_config(page_title="Predictive Lab Throughput", layout="wide")
st.title("🔮 Predictive Centrifuge Health Dashboard")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def compute_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["remaining_hours"] = (df["mtbf_hours"] - df["hours_used"]).clip(lower=0)
    df["days_to_failure"] = df["remaining_hours"] / df["daily_usage_hours"].clip(lower=0.1)

    base_risk = 1 - (df["days_to_failure"] / 30).clip(lower=0, upper=1)
    health_penalty = (df["vibration_score"] * 0.2) + (df["temp_deviation"] * 0.2)
    df["failure_risk"] = (base_risk + health_penalty).clip(lower=0, upper=1)

    def decide_action(row):
        if row["days_to_failure"] < 3 or row["failure_risk"] > 0.8:
            return "Schedule maintenance ASAP"
        elif row["days_to_failure"] < 7:
            return "Plan maintenance this week"
        else:
            return "Monitor only"

    df["recommended_action"] = df.apply(decide_action, axis=1)
    return df

# Load and compute
df_raw = load_data(DATA_PATH)
df = compute_predictions(df_raw)

# Filters / controls
with st.sidebar:
    st.header("Filters")
    max_days = st.slider("Max days to failure (focus on risky equipment)", 1, 60, 30)
    min_risk = st.slider("Min failure risk threshold", 0.0, 1.0, 0.3)
    show_table = st.checkbox("Show raw data table", value=True)

filtered = df[(df["days_to_failure"] <= max_days) | (df["failure_risk"] >= min_risk)]

# Top-level KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total centrifuges", len(df))
col2.metric("High-risk centrifuges", (df["failure_risk"] > 0.8).sum())
col3.metric("Avg days to failure", f"{df['days_to_failure'].mean():.1f}")

st.subheader("Centrifuge Risk Overview")
st.bar_chart(filtered.set_index("name")[["days_to_failure"]])

st.subheader("Detailed Predictions")
st.dataframe(
    filtered[
        [
            "name",
            "days_to_failure",
            "failure_risk",
            "recommended_action",
            "hours_used",
            "daily_usage_hours",
            "vibration_score",
            "temp_deviation",
        ]
    ].sort_values("failure_risk", ascending=False),
    use_container_width=True,
)

if show_table:
    st.subheader("Raw Input Data")
    st.dataframe(df_raw, use_container_width=True)
