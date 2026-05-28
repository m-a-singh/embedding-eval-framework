import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(layout="wide", page_title="Embedding Evaluation Dashboard")

st.title("📊 Embedding Evaluation Framework Data Viewer")


# --- Helper Logic for Data Loading ---
def load_csv_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def load_tsv_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except FileNotFoundError:
        return pd.DataFrame()


# --- Read underlying evaluation dataset to capture metadata fields dynamically ---
sample_path = Path("data/sample.jsonl")
sample_rows: list[dict[str, Any]] = []
detected_fields: list[str] = [
    "name",
    "description",
    "tagCategories",
    "tagTopics",
    "sponsors",
]

if sample_path.exists():
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample_rows.append(json.loads(line))

        found_keys: set[str] = set()
        for r in sample_rows:
            jd = r.get("json_data")
            if isinstance(jd, dict):
                for k in jd.keys():
                    found_keys.add(str(k))
        if found_keys:
            detected_fields = sorted(list(found_keys))
    except Exception:
        pass

# --- Sidebar: Bring Your Own Field Weights Manager ---
st.sidebar.header("⚖️ Strategy Field Weights")
st.sidebar.write("Customize your active importance configurations:")

# Codebase default fallbacks
default_weights = {
    "name": 0.25,
    "description": 0.15,
    "tagCategories": 0.25,
    "tagTopics": 0.25,
    "sponsors": 0.10,
}

# Collect weights into a standard dictionary via sliders
custom_weights: dict[str, float] = {}
for field in detected_fields:
    fallback_val = default_weights.get(field, 0.20)
    custom_weights[field] = st.sidebar.slider(
        f"Weight: `{field}`",
        min_value=0.0,
        max_value=1.0,
        value=float(fallback_val),
        step=0.05,
    )

# Visual validation warning to user if weights are unbalanced
total_weight = sum(custom_weights.values())
if abs(total_weight - 1.0) > 0.001:
    st.sidebar.warning(
        f"⚠️ Weights combine to **{total_weight:.2f}** (Recommended: 1.00)."
    )
else:
    st.sidebar.success("✅ Weights balanced perfectly (Sum: 1.00).")

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Run Model Execution Pipeline")
st.sidebar.write(
    "Trigger calculations over your sample records using your weight modifications above."
)

# --- The Action Button ---
if st.sidebar.button("⚙️ Execute Model Evaluation", type="primary"):
    # 1. Export weights configuration to disk so weighted_chunking.py can read it dynamically
    config_path = Path("data/active_weights.json")
    with open(config_path, "w", encoding="utf-8") as conf_f:
        json.dump(custom_weights, conf_f, indent=4)

    # 2. Run your underlying ecosystem scripts using subprocessing hooks
    with st.spinner(
        "Step 1/2: Compiling forward pass vector embeddings via Triton simulator..."
    ):
        eval_proc = subprocess.run(
            [
                "python3",
                "run_evaluation.py",
                "--data",
                "data/sample.jsonl",
                "--out",
                "results/results.tsv",
            ],
            capture_output=True,
            text=True,
        )

    if eval_proc.returncode != 0:
        st.error(f"Execution Error inside run_evaluation.py:\n{eval_proc.stderr}")
    else:
        with st.spinner(
            "Step 2/2: Recalculating Pearson Correlation matrices & reports..."
        ):
            report_proc = subprocess.run(
                [
                    "python3",
                    "build_report.py",
                    "--in",
                    "results/results.tsv",
                    "--out",
                    "results/summary.csv",
                ],
                capture_output=True,
                text=True,
            )

        if report_proc.returncode != 0:
            st.error(f"Execution Error inside build_report.py:\n{report_proc.stderr}")
        else:
            st.toast(
                "🎉 Pipeline Execution Complete! Dashboard data reloaded successfully.",
                icon="✅",
            )
            # Clear caches to force UI data grid update instantly
            st.cache_data.clear()

# --- Load and Render Updated Data Components ---
summary_df = load_csv_data("results/summary.csv")
results_df = load_tsv_data("results/results.tsv")

# --- Summary View (summary.csv) ---
st.header("Summary of Pearson Correlation Scores")

if not summary_df.empty:
    correlation_columns = [
        col
        for col in summary_df.columns
        if "pearson" in col.lower() or "correlation" in col.lower()
    ]

    if not correlation_columns:
        st.warning(
            "No columns identified as Pearson correlation scores in summary.csv."
        )
        st.dataframe(summary_df)
    else:
        selected_correlation_column = st.selectbox(
            "Select Target Evaluation Correlation Metric:", correlation_columns
        )

        if "chunking_strategy" in summary_df.columns:
            strategies = [
                str(s) for s in summary_df["chunking_strategy"].unique().tolist()
            ]
            selected_strategy = st.selectbox(
                "Filter Data View by Text Formatting Strategy:",
                options=["All"] + strategies,
            )

            if selected_strategy == "All":
                display_df = summary_df
            else:
                display_df = summary_df[
                    summary_df["chunking_strategy"] == selected_strategy
                ]

            if not display_df.empty:
                fig = px.bar(
                    display_df,
                    x="chunking_strategy",
                    y=selected_correlation_column,
                    color="chunking_strategy",
                    title=f"Pearson Correlation Scores Comparison Chart ({selected_correlation_column})",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(display_df)
            else:
                st.info(
                    "No tracking row elements available to display matching filtered target strategy criteria."
                )
        else:
            st.bar_chart(summary_df[selected_correlation_column])
            st.dataframe(summary_df)
else:
    st.info(
        "No matching validation statistics are available. Please adjust your weights and click 'Execute Model Evaluation' in the sidebar."
    )

# --- Detailed Evaluation Results View (results.tsv) ---
st.header("Detailed Results Matrix")

if not results_df.empty:
    if "chunking_strategy" in results_df.columns:
        unique_strats = ["All"] + [
            str(s) for s in results_df["chunking_strategy"].unique()
        ]
        filter_strat = st.selectbox(
            "Filter Detailed Matrix View via Strategy:", options=unique_strats
        )
        if filter_strat != "All":
            results_df = results_df[results_df["chunking_strategy"] == filter_strat]

    st.dataframe(results_df)
else:
    st.info("No detailed evaluation logs detected on disk.")
