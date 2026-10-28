
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "models" / "customer_segmentation.joblib"

# Example exchange rate: change this when needed.
# This assumes Annual Income (k$) means thousands of US dollars.
USD_TO_INR = 88.0

INCOME_MULTIPLIER = 1000 * USD_TO_INR

INCOME_COL = "Annual Income (k$)"
SPENDING_COL = "Spending Score (1-100)"

CLUSTER_COLUMN_MAP = {
    "K-Means": "KMeans_Cluster",
    "Agglomerative": "Agglomerative_Cluster",
    "DBSCAN": "DBSCAN_Cluster",
    "Gaussian Mixture": "Gaussian_Mixture_Cluster",
    "BIRCH": "BIRCH_Cluster",
    "Mean Shift": "Mean_Shift_Cluster",
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_inr(value):
    """Format a number as Indian rupees."""
    return f"₹{value:,.0f}"


def income_kusd_to_inr(income_kusd):
    """Convert income in thousands of USD to INR."""
    return float(income_kusd) * INCOME_MULTIPLIER


def income_inr_to_kusd(income_inr):
    """Convert INR to the model's original income unit."""
    return float(income_inr) / INCOME_MULTIPLIER


def get_cluster_column(algorithm, clustered_data):
    """Find the cluster column associated with an algorithm."""
    expected = CLUSTER_COLUMN_MAP.get(algorithm)

    if expected and expected in clustered_data.columns:
        return expected

    fallback = algorithm.replace(" ", "_") + "_Cluster"

    if fallback in clustered_data.columns:
        return fallback

    raise KeyError(
        f"Cluster column for '{algorithm}' was not found. "
        f"Available columns: {list(clustered_data.columns)}"
    )


def describe_segment(
    income,
    spending,
    income_mid,
    spending_mid,
    income_low,
    income_high,
    spending_low,
    spending_high,
):
    """
    Describe the customer using the original training-data units.
    Income is in thousands of USD here, not INR.
    """

    if income >= income_mid and spending >= spending_mid:
        return (
            "High income, high spending",
            "Premium Shoppers",
            "Premium retail events and loyalty benefits"
        )

    if income < income_mid and spending >= spending_mid:
        return (
            "Low income, high spending",
            "Active Spenders",
            "Value-added rewards and repeat-visit incentives"
        )

    if income >= income_mid and spending < spending_mid:
        return (
            "High income, low spending",
            "Growth Opportunity",
            "Relevant premium experiences and targeted offers"
        )

    if income < income_mid and spending < spending_mid:
        return (
            "Low income, low spending",
            "Value Seekers",
            "Discount days and affordable shopping promotions"
        )

    return (
        "Middle-range customer",
        "Regular Shoppers",
        "Seasonal campaigns and category-specific promotions"
    )


def predict_customer(artifact, algorithm, income_inr, spending):
    """
    Predict a customer's cluster using the saved model.

    Models with predict() use their trained prediction method.
    Models without predict(), such as Agglomerative or DBSCAN,
    use the nearest cluster centroid in the saved scaled feature space.
    """

    features = artifact["features"]
    scaler = artifact["scaler"]
    models = artifact["models"]
    clustered = artifact["clustered"]

    income_kusd = income_inr_to_kusd(income_inr)

    # Keep the same feature order and original units as training.
    customer = pd.DataFrame(
        [[income_kusd, spending]],
        columns=features
    )

    customer_scaled = scaler.transform(customer)

    model = models[algorithm]

    # Models that support predicting unseen observations.
    if callable(getattr(model, "predict", None)):
        try:
            cluster = int(model.predict(customer_scaled)[0])
            method = "Model prediction"
            return cluster, method, income_kusd
        except (AttributeError, NotImplementedError):
            pass

    # Fallback for models that do not support predict().
    # Estimate the closest cluster using scaled training observations.
    cluster_col = get_cluster_column(algorithm, clustered)

    valid_rows = clustered[cluster_col].notna()
    training = clustered.loc[valid_rows].copy()

    if training.empty:
        raise ValueError(
            f"No valid training cluster assignments found for {algorithm}."
        )

    training_features = training[features]
    training_scaled = scaler.transform(training_features)

    labels = training[cluster_col].to_numpy()

    # DBSCAN's -1 label means noise. Exclude it from centroid matching
    # when other clusters are available.
    non_noise = labels != -1

    if non_noise.any():
        training_scaled = training_scaled[non_noise]
        labels = labels[non_noise]

    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        points = training_scaled[labels == label]
        centroids.append(points.mean(axis=0))

    centroids = np.asarray(centroids)

    distances = np.linalg.norm(
        centroids - customer_scaled[0],
        axis=1
    )

    nearest_index = int(np.argmin(distances))
    cluster = int(unique_labels[nearest_index])

    return cluster, "Nearest cluster centroid estimate", income_kusd


def safe_metrics(artifact, algorithm):
    """Return the metrics for an algorithm when available."""
    metrics = artifact.get("metrics", {})

    if isinstance(metrics, pd.DataFrame):
        if algorithm in metrics.index:
            return metrics.loc[algorithm].to_dict()
        return {}

    if isinstance(metrics, dict):
        item = metrics.get(algorithm, {})
        return item if isinstance(item, dict) else {}

    return {}


# =========================================================
# LOAD TRAINED ARTIFACT
# =========================================================

@st.cache_resource
def load_artifact(path_string):
    return joblib.load(path_string)


st.title("🛍️ Customer Segmentation Dashboard")
st.caption(
    "Analyze customer profiles using your previously trained "
    "segmentation models."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Check that customer_segmentation.joblib is inside the "
        "project's models folder. Run train_model.py if you have "
        "not generated the model artifact yet."
    )
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


required_keys = ["scaler", "models", "features", "clustered"]

missing_keys = [key for key in required_keys if key not in artifact]

if missing_keys:
    st.error(f"The model artifact is missing these keys: {missing_keys}")
    st.stop()

features = artifact["features"]
clustered = artifact["clustered"]
models = artifact["models"]

if INCOME_COL not in features or SPENDING_COL not in features:
    st.error(
        "The trained model's features do not match the expected columns. "
        f"Found: {features}"
    )
    st.stop()

if clustered.empty:
    st.error("The saved training dataset is empty.")
    st.stop()

if INCOME_COL not in clustered.columns or SPENDING_COL not in clustered.columns:
    st.error("The training data is missing income or spending columns.")
    st.stop()

if not models:
    st.error("No trained models were found in the artifact.")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Settings")

algorithm = st.sidebar.selectbox(
    "Choose segmentation algorithm",
    options=list(models.keys())
)

cluster_col = get_cluster_column(algorithm, clustered)

st.sidebar.caption(
    f"Selected model: {algorithm}"
)

st.sidebar.caption(
    f"Display currency conversion: 1 USD = ₹{USD_TO_INR:.2f}"
)

st.sidebar.caption(
    "Change USD_TO_INR in app2.py to update the display conversion rate."
)


# =========================================================
# PREPARE DATA FOR DISPLAY
# =========================================================

display_df = clustered.copy()

display_df["Annual Income (₹)"] = (
    display_df[INCOME_COL] * INCOME_MULTIPLIER
)

display_df["Displayed Cluster"] = (
    display_df[cluster_col].astype(str)
)

valid_income = clustered[INCOME_COL].dropna()
valid_spending = clustered[SPENDING_COL].dropna()

income_mid = float(valid_income.median())
spending_mid = float(valid_spending.median())

income_low = float(valid_income.quantile(0.40))
income_high = float(valid_income.quantile(0.60))

spending_low = float(valid_spending.quantile(0.40))
spending_high = float(valid_spending.quantile(0.60))


# =========================================================
# SUMMARY METRICS
# =========================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Training Customers", f"{len(clustered):,}")

with metric2:
    st.metric(
        "Unique Clusters",
        f"{clustered[cluster_col].nunique(dropna=True):,}"
    )

with metric3:
    st.metric(
        "Median Annual Income",
        format_inr(income_kusd_to_inr(income_mid))
    )

with metric4:
    st.metric(
        "Median Spending Score",
        f"{spending_mid:.1f}/100"
    )


# =========================================================
# DASHBOARD TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Customer Analysis",
    "Algorithm Comparison",
    "Cluster Insights",
    "Data Explorer",
])


# =========================================================
# TAB 1: LIVE CUSTOMER ANALYSIS
# =========================================================

with tab1:
    st.subheader("Analyze a New Customer")

    st.write(
        "Enter the customer's annual income in Indian rupees and "
        "their spending score. The app converts income to the "
        "training-data units before running the model."
    )

    min_income_inr = max(
        0,
        int(income_kusd_to_inr(valid_income.min()))
    )

    max_income_inr = int(
        income_kusd_to_inr(valid_income.max())
    )

    default_income = int(
        income_kusd_to_inr(income_mid)
    )

    with st.form("customer_analysis_form"):
        col1, col2 = st.columns(2)

        with col1:
            income_inr = st.number_input(
                "Annual income (₹)",
                min_value=0,
                max_value=max(max_income_inr * 3, 100000),
                value=default_income,
                step=10000,
                format="%d",
                help=(
                    "Enter annual income in rupees. It is converted "
                    "to the original model units before prediction."
                )
            )

        with col2:
            spending = st.slider(
                "Spending score (1–100)",
                min_value=1,
                max_value=100,
                value=50
            )

        submitted = st.form_submit_button(
            "Analyze Customer",
            use_container_width=True
        )

    if submitted:
        try:
            predicted_cluster, prediction_method, income_kusd = (
                predict_customer(
                    artifact,
                    algorithm,
                    income_inr,
                    spending
                )
            )

            profile, segment_name, business_action = describe_segment(
                income=income_kusd,
                spending=spending,
                income_mid=income_mid,
                spending_mid=spending_mid,
                income_low=income_low,
                income_high=income_high,
                spending_low=spending_low,
                spending_high=spending_high,
            )

            st.session_state["customer_analysis"] = {
                "income_inr": float(income_inr),
                "income_kusd": float(income_kusd),
                "spending": int(spending),
                "cluster": int(predicted_cluster),
                "profile": profile,
                "segment_name": segment_name,
                "business_action": business_action,
                "algorithm": algorithm,
                "prediction_method": prediction_method,
            }

        except Exception as error:
            st.error(f"Could not analyze this customer: {error}")

    result = st.session_state.get("customer_analysis")

    if result:
        st.divider()
        st.subheader("Customer Analysis Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Annual Income",
                format_inr(result["income_inr"])
            )

        with r2:
            st.metric(
                "Spending Score",
                f'{result["spending"]}/100'
            )

        with r3:
            st.metric(
                "Predicted Cluster",
                str(result["cluster"])
            )

        st.caption(
            f'Model: {result["algorithm"]} · '
            f'Method: {result["prediction_method"]}'
        )

        out1, out2 = st.columns(2)

        with out1:
            st.markdown("#### Customer profile")
            st.success(result["profile"])

            st.markdown("#### Suggested segment name")
            st.info(result["segment_name"])

        with out2:
            st.markdown("#### Possible business action")
            st.write(result["business_action"])

            st.markdown("#### Interpretation")
            st.write(
                "The profile and suggested business action are based "
                "on income and spending thresholds in the training data. "
                "The predicted cluster is produced separately by the "
                "selected segmentation algorithm."
            )

        # Scatter plot with the current customer highlighted.
        plot_df = display_df.copy()

        fig = px.scatter(
            plot_df,
            x="Annual Income (₹)",
            y=SPENDING_COL,
            color="Displayed Cluster",
            title="Customer Segments and Your Customer",
            hover_data=[INCOME_COL, SPENDING_COL],
            labels={
                "Annual Income (₹)": "Annual Income (₹)",
                SPENDING_COL: "Spending Score",
                "Displayed Cluster": "Cluster"
            }
        )

        fig.add_scatter(
            x=[result["income_inr"]],
            y=[result["spending"]],
            mode="markers",
            name="New Customer",
            marker={
                "size": 18,
                "symbol": "star",
                "line": {"width": 2}
            }
        )

        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 2: ALGORITHM COMPARISON
# =========================================================

with tab2:
    st.subheader("Compare Trained Algorithms")

    comparison_rows = []

    for model_name in models:
        metrics = safe_metrics(artifact, model_name)

        row = {"Algorithm": model_name}

        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if np.isscalar(value) or value is None:
                    row[str(key)] = value

        comparison_rows.append(row)

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No saved model evaluation metrics are available.")

    if "elbow" in artifact:
        elbow = artifact["elbow"]

        if isinstance(elbow, dict) and elbow:
            elbow_df = pd.DataFrame({
                "Number of Clusters": list(elbow.keys()),
                "Inertia": list(elbow.values())
            })

            elbow_df["Number of Clusters"] = pd.to_numeric(
                elbow_df["Number of Clusters"],
                errors="coerce"
            )
            elbow_df["Inertia"] = pd.to_numeric(
                elbow_df["Inertia"],
                errors="coerce"
            )
            elbow_df = elbow_df.dropna()

            if not elbow_df.empty:
                fig_elbow = px.line(
                    elbow_df.sort_values("Number of Clusters"),
                    x="Number of Clusters",
                    y="Inertia",
                    markers=True,
                    title="Elbow Method"
                )
                st.plotly_chart(
                    fig_elbow,
                    use_container_width=True
                )

    st.caption(
        "Metrics are shown only when they were saved in the model artifact."
    )


# =========================================================
# TAB 3: CLUSTER INSIGHTS
# =========================================================

with tab3:
    st.subheader(f"Cluster Insights — {algorithm}")

    summary = (
        display_df.groupby("Displayed Cluster", dropna=False)
        .agg(
            Customers=(SPENDING_COL, "size"),
            Mean_Income_INR=("Annual Income (₹)", "mean"),
            Median_Income_INR=("Annual Income (₹)", "median"),
            Mean_Spending=(SPENDING_COL, "mean"),
            Median_Spending=(SPENDING_COL, "median"),
        )
        .reset_index()
    )

    summary["Mean Annual Income (₹)"] = summary[
        "Mean_Income_INR"
    ].map(format_inr)

    summary["Median Annual Income (₹)"] = summary[
        "Median_Income_INR"
    ].map(format_inr)

    summary["Mean Spending Score"] = summary[
        "Mean_Spending"
    ].round(2)

    summary["Median Spending Score"] = summary[
        "Median_Spending"
    ].round(2)

    summary_display = summary[
        [
            "Displayed Cluster",
            "Customers",
            "Mean Annual Income (₹)",
            "Median Annual Income (₹)",
            "Mean Spending Score",
            "Median Spending Score",
        ]
    ]

    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:
        fig_income = px.bar(
            summary,
            x="Displayed Cluster",
            y="Mean_Income_INR",
            title="Average Annual Income by Cluster",
            labels={
                "Displayed Cluster": "Cluster",
                "Mean_Income_INR": "Average Annual Income (₹)"
            }
        )
        st.plotly_chart(fig_income, use_container_width=True)

    with right:
        fig_spending = px.bar(
            summary,
            x="Displayed Cluster",
            y="Mean_Spending",
            title="Average Spending Score by Cluster",
            labels={
                "Displayed Cluster": "Cluster",
                "Mean_Spending": "Average Spending Score"
            }
        )
        st.plotly_chart(fig_spending, use_container_width=True)

    fig_distribution = px.pie(
        summary,
        names="Displayed Cluster",
        values="Customers",
        title="Customer Distribution by Cluster"
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True
    )


# =========================================================
# TAB 4: DATA EXPLORER
# =========================================================

with tab4:
    st.subheader("Explore Training Customers")

    min_display_income = int(
        max(0, income_kusd_to_inr(valid_income.min()))
    )
    max_display_income = int(
        income_kusd_to_inr(valid_income.max())
    )

    income_range = st.slider(
        "Filter by annual income (₹)",
        min_value=min_display_income,
        max_value=max(max_display_income, min_display_income + 1),
        value=(
            min_display_income,
            max(max_display_income, min_display_income + 1)
        ),
        step=1000
    )

    min_spending, max_spending = st.slider(
        "Filter by spending score",
        min_value=1,
        max_value=100,
        value=(1, 100)
    )

    filtered = display_df[
        display_df["Annual Income (₹)"].between(
            income_range[0],
            income_range[1]
        )
        & display_df[SPENDING_COL].between(
            min_spending,
            max_spending
        )
    ].copy()

    st.write(f"Matching customers: **{len(filtered):,}**")

    columns_to_show = [
        col for col in [
            INCOME_COL,
            "Annual Income (₹)",
            SPENDING_COL,
            cluster_col
        ]
        if col in filtered.columns
    ]

    table_df = filtered[columns_to_show].copy()

    if INCOME_COL in table_df.columns:
        table_df = table_df.drop(columns=[INCOME_COL])

    if "Annual Income (₹)" in table_df.columns:
        table_df["Annual Income (₹)"] = (
            table_df["Annual Income (₹)"].round(0).astype("Int64")
        )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )

    csv = table_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="customer_segments_inr.csv",
        mime="text/csv"
    )
