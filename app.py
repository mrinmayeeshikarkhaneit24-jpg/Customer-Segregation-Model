
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")
st.write(
    "Discover customer groups using "
    "K-Means clustering and spending patterns."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Mall_Customers.csv was not found.")
    st.stop()

# Select features used by the model
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

if not all(col in df.columns for col in features):
    st.error("The dataset does not contain the required columns.")
    st.stop()

X = df[features].dropna()

# Sidebar settings
st.sidebar.header("Model Settings")
k = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=10,
    value=5
)

# Train K-Means
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X)

result = df.loc[X.index].copy()
result["Cluster"] = labels.astype(str)

score = silhouette_score(X, labels)

# Summary cards
c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", len(result))
c2.metric("Number of Clusters", k)
c3.metric("Silhouette Score", f"{score:.3f}")

st.divider()

# Cluster visualization
st.subheader("Customer Segments")

fig = px.scatter(
    result,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster",
    hover_data=["CustomerID", "Gender", "Age"],
    title="Annual Income vs Spending Score",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Dataset explorer
st.subheader("Customer Data Explorer")

search = st.text_input(
    "Search by customer ID or gender"
)

if search:
    result_view = result[
        result["CustomerID"].astype(str).str.contains(
            search, case=False, na=False
        )
        | result["Gender"].astype(str).str.contains(
            search, case=False, na=False
        )
    ]
else:
    result_view = result

st.dataframe(result_view, use_container_width=True)

# Download clustered data
csv = result.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Segmentation Results",
    data=csv,
    file_name="Customer_Segmentation_Results.csv",
    mime="text/csv"
)


# Elbow Method and Silhouette Analysis
st.divider()
st.header("Model Evaluation")
st.write(
    "Compare different values of K to understand "
    "the number of clusters suitable for your dataset."
)

max_k = min(10, len(X) - 1)

if max_k >= 3:
    wcss = []
    silhouette_scores = []

    k_values = list(range(2, max_k + 1))

    with st.spinner("Evaluating different cluster counts..."):
        for i in k_values:
            temp_model = KMeans(
                n_clusters=i,
                random_state=42,
                n_init=10
            )

            temp_labels = temp_model.fit_predict(X)

            # Within-cluster sum of squares
            wcss.append(temp_model.inertia_)

            # Cluster separation and cohesion
            silhouette_scores.append(
                silhouette_score(X, temp_labels)
            )

    col1, col2 = st.columns(2)

    # Elbow Method chart
    with col1:
        st.subheader("1. Elbow Method")

        elbow_df = pd.DataFrame({
            "Number of Clusters (K)": k_values,
            "WCSS": wcss
        })

        elbow_fig = px.line(
            elbow_df,
            x="Number of Clusters (K)",
            y="WCSS",
            markers=True,
            title="WCSS vs Number of Clusters",
            template="plotly_white"
        )

        elbow_fig.update_layout(
            xaxis=dict(dtick=1)
        )

        st.plotly_chart(
            elbow_fig,
            use_container_width=True
        )

        st.caption(
            "Look for the bend where the decrease in WCSS "
            "starts becoming less pronounced."
        )

    # Silhouette Score chart
    with col2:
        st.subheader("2. Silhouette Score")

        silhouette_df = pd.DataFrame({
            "Number of Clusters (K)": k_values,
            "Silhouette Score": silhouette_scores
        })

        silhouette_fig = px.line(
            silhouette_df,
            x="Number of Clusters (K)",
            y="Silhouette Score",
            markers=True,
            title="Silhouette Score vs K",
            template="plotly_white"
        )

        silhouette_fig.update_layout(
            xaxis=dict(dtick=1),
            yaxis=dict(
                range=[
                    min(-1, min(silhouette_scores) - 0.1),
                    min(1, max(silhouette_scores) + 0.1)
                ]
            )
        )

        st.plotly_chart(
            silhouette_fig,
            use_container_width=True
        )

        best_index = silhouette_scores.index(
            max(silhouette_scores)
        )

        st.metric(
            "Highest Silhouette Score",
            f"{silhouette_scores[best_index]:.3f}",
            f"At K = {k_values[best_index]}"
        )

    st.info(
        f"Your current dashboard uses K = {k}. "
        f"The highest silhouette score in the tested range "
        f"occurs at K = {k_values[best_index]}. "
        "Compare both charts and the business usefulness "
        "of the segments before choosing the final K."
    )

else:
    st.warning(
        "Not enough customer records to compare cluster counts."
    )


# Cluster profile

# Cluster Analysis and Business Insights
st.divider()
st.header("Customer Cluster Analysis")

# Calculate cluster statistics
cluster_summary = result.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
    Avg_Age=("Age", "mean")
).reset_index()

cluster_summary["Percentage"] = (
    cluster_summary["Customers"] / len(result) * 100
).round(1)

cluster_summary["Avg_Income"] = (
    cluster_summary["Avg_Income"].round(2)
)

cluster_summary["Avg_Spending"] = (
    cluster_summary["Avg_Spending"].round(2)
)

cluster_summary["Avg_Age"] = (
    cluster_summary["Avg_Age"].round(1)
)

# Generate descriptive segment names from relative averages
income_mid = result["Annual Income (k$)"].median()
spending_mid = result["Spending Score (1-100)"].median()

def describe_cluster(row):
    high_income = row["Avg_Income"] >= income_mid
    high_spending = row["Avg_Spending"] >= spending_mid

    if high_income and high_spending:
        return "High Income, High Spending"
    elif high_income and not high_spending:
        return "High Income, Low Spending"
    elif not high_income and high_spending:
        return "Lower Income, High Spending"
    else:
        return "Lower Income, Low Spending"

cluster_summary["Segment"] = cluster_summary.apply(
    describe_cluster,
    axis=1
)

# Distribution chart
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Distribution")

    distribution_fig = px.pie(
        cluster_summary,
        names="Cluster",
        values="Customers",
        hole=0.45,
        title="Customers in Each Cluster",
        template="plotly_white"
    )

    st.plotly_chart(
        distribution_fig,
        use_container_width=True
    )

with col2:
    st.subheader("Average Spending by Cluster")

    spending_fig = px.bar(
        cluster_summary,
        x="Cluster",
        y="Avg_Spending",
        color="Cluster",
        text="Avg_Spending",
        title="Mean Spending Score",
        labels={
            "Cluster": "Cluster",
            "Avg_Spending": "Average Spending Score"
        },
        template="plotly_white"
    )

    st.plotly_chart(
        spending_fig,
        use_container_width=True
    )

# Detailed profile table
st.subheader("Cluster Profiles")

display_summary = cluster_summary.rename(columns={
    "Cluster": "Cluster ID",
    "Customers": "Customer Count",
    "Percentage": "Customer Share (%)",
    "Avg_Income": "Average Income (k$)",
    "Avg_Spending": "Average Spending Score",
    "Avg_Age": "Average Age",
    "Segment": "Segment Description"
})

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)

# Business recommendations
st.subheader("Business Recommendations")

for _, row in cluster_summary.iterrows():
    with st.expander(
        f"Cluster {row['Cluster']} — {row['Segment']}"
    ):
        st.write(
            f"**Customer count:** {int(row['Customers'])} "
            f"({row['Percentage']}% of the dataset)"
        )

        st.write(
            f"**Average annual income:** "
            f"${row['Avg_Income']}k"
        )

        st.write(
            f"**Average spending score:** "
            f"{row['Avg_Spending']} / 100"
        )

        st.write(
            f"**Average age:** {row['Avg_Age']} years"
        )

        if (
            row["Avg_Income"] >= income_mid
            and row["Avg_Spending"] >= spending_mid
        ):
            recommendation = (
                "Consider premium loyalty benefits, exclusive "
                "offers and personalized rewards."
            )
        elif (
            row["Avg_Income"] >= income_mid
            and row["Avg_Spending"] < spending_mid
        ):
            recommendation = (
                "Test targeted promotions and product bundles "
                "to understand what may encourage purchases."
            )
        elif (
            row["Avg_Income"] < income_mid
            and row["Avg_Spending"] >= spending_mid
        ):
            recommendation = (
                "Consider affordable loyalty rewards and "
                "relevant offers that encourage repeat purchases."
            )
        else:
            recommendation = (
                "Consider value bundles, entry-level products "
                "and carefully tested discounts."
            )

        st.info(recommendation)


# Individual Customer Explorer
st.divider()
st.header("Customer Explorer")
st.write("Search for a customer to explore their profile and cluster.")

# Select customer
customer_ids = result["CustomerID"].dropna().unique().tolist()

selected_id = st.selectbox(
    "Select Customer ID",
    options=sorted(customer_ids)
)

customer = result[
    result["CustomerID"] == selected_id
].iloc[0]

cluster_id = str(customer["Cluster"])

# Find the corresponding segment description
segment_row = cluster_summary[
    cluster_summary["Cluster"] == cluster_id
].iloc[0]

# Customer metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("Customer ID", int(customer["CustomerID"]))
c2.metric("Age", f"{int(customer['Age'])} years")
c3.metric("Annual Income", f"${customer['Annual Income (k$)']}k")
c4.metric(
    "Spending Score",
    f"{int(customer['Spending Score (1-100)'])}/100"
)

st.subheader("Customer Profile")

left, right = st.columns(2)

with left:
    st.write(f"**Gender:** {customer['Gender']}")
    st.write(f"**Assigned Cluster:** {cluster_id}")
    st.write(f"**Segment:** {segment_row['Segment']}")

with right:
    st.write("**Segment averages**")
    st.write(
        f"Average income: ${segment_row['Avg_Income']}k"
    )
    st.write(
        f"Average spending score: "
        f"{segment_row['Avg_Spending']}/100"
    )

# Highlight the selected customer
st.subheader("Customer Position in Segmentation")

highlight_fig = px.scatter(
    result,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster",
    opacity=0.35,
    hover_data=["CustomerID", "Gender", "Age"],
    template="plotly_white",
    title="Selected Customer Among All Segments"
)

highlight_fig.add_scatter(
    x=[customer["Annual Income (k$)"]],
    y=[customer["Spending Score (1-100)"]],
    mode="markers+text",
    marker=dict(
        size=18,
        color="black",
        symbol="star",
        line=dict(width=2, color="white")
    ),
    text=[f"Customer {int(selected_id)}"],
    textposition="top center",
    name="Selected Customer"
)

st.plotly_chart(
    highlight_fig,
    use_container_width=True
)

