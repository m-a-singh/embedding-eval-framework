
import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(layout="wide")

st.title("Embedding Evaluation Framework Data Viewer")

# --- Load Data ---
@st.cache_data
def load_csv_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {path}")
        return pd.DataFrame()

@st.cache_data
def load_tsv_data(path):
    try:
        return pd.read_csv(path, sep='\t')
    except FileNotFoundError:
        st.error(f"Error: File not found at {path}")
        return pd.DataFrame()

summary_df = load_csv_data("results/summary.csv")
results_df = load_tsv_data("results/results.tsv")

# --- Summary View (summary.csv) ---
st.header("Summary of Pearson Correlation Scores")

if not summary_df.empty:
    # Assuming summary.csv has columns like 'strategy' and 'pearson_correlation'
    # If the column names are different, please adjust them here.

    # Identify columns that likely contain Pearson correlation scores.
    # This is a heuristic and might need adjustment based on actual column names.
    correlation_columns = [col for col in summary_df.columns if "pearson" in col.lower() or "correlation" in col.lower()]

    if not correlation_columns:
        st.warning("No columns identified as Pearson correlation scores in summary.csv. Displaying raw data.")
        st.dataframe(summary_df)
    else:
        # For simplicity, let's assume the first identified correlation column is the primary one.
        # If there are multiple, you might want to let the user select or display all.
        selected_correlation_column = correlation_columns[0]

        st.subheader("Pearson Correlation Scores by Strategy")

        # Dropdown for text formatting strategies
        if 'chunking_strategy' in summary_df.columns:
            strategies = summary_df['chunking_strategy'].unique().tolist()
            selected_strategy = st.selectbox(
                "Select Text Formatting Strategy (for comparison)",
                options=["All"] + strategies
            )

            if selected_strategy == "All":
                display_df = summary_df
            else:
                display_df = summary_df[summary_df['chunking_strategy'] == selected_strategy]

            if not display_df.empty:
                st.bar_chart(display_df.set_index('chunking_strategy')[selected_correlation_column])
                st.dataframe(display_df)
            else:
                st.info("No data to display for the selected strategy.")
        else:
            st.warning("No 'chunking_strategy' column found in summary.csv. Displaying correlation scores directly.")
            st.bar_chart(summary_df[selected_correlation_column])
            st.dataframe(summary_df)
else:
    st.info("No data available in summary.csv to display.")

# --- Results View (results.tsv) ---
st.header("Detailed Results")

if not results_df.empty:
    st.dataframe(results_df)
else:
    st.info("No data available in results.tsv to display.")
