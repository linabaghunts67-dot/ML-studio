import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import get_scenario_config, load_data
from src.data_prep import basic_clean
from src.evaluation import train_and_evaluate_models

st.set_page_config(
    page_title="TabularML Studio",
    layout="wide",
)

SCENARIOS = {
    "Price Prediction (Phones)": "price",
    "Credit Default Risk": "default",
    "Customer Churn": "churn",
}

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

if "last_test_df" not in st.session_state:
    st.session_state.last_test_df = None


def main():
    st.sidebar.title("TabularML Studio")
    st.sidebar.markdown("Choose a scenario to start:")

    scenario_label = st.sidebar.radio(
        "Scenario",
        list(SCENARIOS.keys()),
        index=0,
    )
    scenario_id = SCENARIOS[scenario_label]
    cfg = get_scenario_config(scenario_id)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Current scenario: `{cfg.id}`")
    st.sidebar.markdown(f"Task type: `{cfg.task_type}`")
    st.sidebar.markdown(f"Target column: `{cfg.target}`")

    df_raw = load_data(scenario_id)

    st.title("TabularML Studio 🧠📊")
    st.subheader("Interactive data cleaning and modeling playground")
    st.markdown(
        f"### Scenario: {cfg.name}\n"
        "Steps: **Overview → Clean → Explore → Model → Download**."
    )

    tab_overview, tab_clean, tab_explore, tab_model, tab_download = st.tabs(
        ["1. Data overview", "2. Clean & fill", "3. Explore", "4. Train models", "5. Download"]
    )

    with tab_overview:
        st.header("1. Data overview")
        st.dataframe(df_raw.head())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df_raw))
        with col2:
            st.metric("Columns", df_raw.shape[1])
        with col3:
            st.metric("Missing values (total)", int(df_raw.isna().sum().sum()))

        st.write(df_raw.isna().sum())
        st.write(df_raw.select_dtypes(include="number").describe())

    with tab_clean:
        st.header("2. Clean & fill the data")

        numeric_strategy = st.selectbox(
            "Numeric missing value strategy",
            ["mean", "median", "drop rows with NaNs"],
            index=0,
        )
        categorical_strategy = st.selectbox(
            "Categorical missing value strategy",
            ["most_frequent", "create 'Unknown' category"],
            index=0,
        )

        cap_outliers = st.checkbox("Cap numeric outliers at 1st and 99th percentile", value=False)

        if st.button("Apply cleaning"):
            df_clean = basic_clean(
                df_raw,
                target_col=cfg.target,
                numeric_strategy=numeric_strategy,
                categorical_strategy=categorical_strategy,
                cap_outliers=cap_outliers,
            )
            st.session_state.df_clean = df_clean
            st.dataframe(df_clean.head())

            before_missing = df_raw.isna().sum().sum()
            after_missing = df_clean.isna().sum().sum()
            st.markdown(f"- Missing values before: **{before_missing}**")
            st.markdown(f"- Missing values after: **{after_missing}**")

    df_for_analysis = (
        st.session_state.df_clean if st.session_state.df_clean is not None else df_raw
    )

    with tab_explore:
        st.header("3. Explore & visualize")

        feature_cols = [c for c in df_for_analysis.columns if c != cfg.target]

        if feature_cols:
            feature = st.selectbox("Select a feature", feature_cols)

            col_left, col_right = st.columns(2)

            with col_left:
                if df_for_analysis[feature].dtype == "O":
                    fig = px.bar(
                        df_for_analysis[feature].value_counts().reset_index(),
                        x="index",
                        y=feature,
                        labels={"index": feature, feature: "Count"},
                    )
                    st.plotly_chart(fig)
                else:
                    fig = px.histogram(
                        df_for_analysis,
                        x=feature,
                        nbins=40,
                    )
                    st.plotly_chart(fig)

            with col_right:
                try:
                    fig2 = px.scatter(
                        df_for_analysis,
                        x=feature,
                        y=cfg.target,
                    )
                    st.plotly_chart(fig2)
                except Exception:
                    st.warning("Scatter plot not available for this combination.")
        else:
            st.warning("No feature columns available.")

    with tab_model:
        st.header("4. Train and evaluate models")

        selected_model_names = []

        if cfg.task_type == "regression":
            if st.checkbox("Linear Regression", value=True):
                selected_model_names.append("linear")
            if st.checkbox("Random Forest Regressor", value=False):
                selected_model_names.append("rf")
        else:
            if st.checkbox("Logistic Regression", value=True):
                selected_model_names.append("log_reg")
            if st.checkbox("Random Forest Classifier", value=False):
                selected_model_names.append("rf")

        n_estimators = st.slider(
            "Number of trees for Random Forest", 10, 300, 100, 10
        )

        if st.button("Train & evaluate"):
            if selected_model_names:
                metrics_df, fitted_models, test_df = train_and_evaluate_models(
                    df_for_analysis,
                    target_col=cfg.target,
                    task_type=cfg.task_type,
                    model_names=selected_model_names,
                    n_estimators=n_estimators,
                )
                st.session_state.last_metrics = metrics_df
                st.session_state.last_test_df = test_df
                st.dataframe(metrics_df)
            else:
                st.error("Please select at least one model.")

    with tab_download:
        st.header("5. Download results")

        if st.session_state.df_clean is not None:
            clean_csv = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download cleaned data (CSV)",
                data=clean_csv,
                file_name=f"{cfg.id}_cleaned.csv",
                mime="text/csv",
            )

        if st.session_state.last_test_df is not None:
            pred_csv = st.session_state.last_test_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download test set with predictions (CSV)",
                data=pred_csv,
                file_name=f"{cfg.id}_test_with_predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
