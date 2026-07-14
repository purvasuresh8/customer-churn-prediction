import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/predictions/retention_plan.csv"
    )

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["RiskLevel"].unique(),
    default=df["RiskLevel"].unique()
)

filtered_df = df[
    df["RiskLevel"].isin(risk_filter)
]

# =====================================================
# HEADER
# =====================================================

st.title("📊 Customer Churn Prediction & Retention Analytics Platform")

st.markdown(
    """
    This dashboard provides:
    - Churn Prediction Analytics
    - Customer Risk Segmentation
    - Retention Recommendations
    - Business Insights
    """
)

# =====================================================
# KPIs
# =====================================================

total_customers = len(filtered_df)

high_risk = len(
    filtered_df[
        filtered_df["RiskLevel"] == "High"
    ]
)

avg_churn_prob = round(
    filtered_df["ChurnProbability"].mean() * 100,
    2
)

avg_priority = round(
    filtered_df["PriorityScore"].mean(),
    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    total_customers
)

col2.metric(
    "High Risk Customers",
    high_risk
)

col3.metric(
    "Average Churn Probability %",
    avg_churn_prob
)

col4.metric(
    "Average Priority Score",
    avg_priority
)

# =====================================================
# RISK DISTRIBUTION
# =====================================================

st.subheader("Customer Risk Distribution")

risk_chart = px.histogram(
    filtered_df,
    x="RiskLevel",
    color="RiskLevel",
    title="Risk Level Breakdown"
)

st.plotly_chart(
    risk_chart,
    use_container_width=True
)

# =====================================================
# CHURN PROBABILITY DISTRIBUTION
# =====================================================

st.subheader("Churn Probability Distribution")

prob_chart = px.histogram(
    filtered_df,
    x="ChurnProbability",
    nbins=25,
    title="Predicted Churn Probability"
)

st.plotly_chart(
    prob_chart,
    use_container_width=True
)

# =====================================================
# TOP AT-RISK CUSTOMERS
# =====================================================

st.subheader("Top At-Risk Customers")

top_risk = filtered_df.sort_values(
    by="ChurnProbability",
    ascending=False
).head(20)

st.dataframe(
    top_risk,
    use_container_width=True
)

# =====================================================
# PRIORITY SCORE CHART
# =====================================================

st.subheader("Priority Scores")

priority_chart = px.scatter(
    filtered_df,
    x="PriorityScore",
    y="ChurnProbability",
    color="RiskLevel",
    hover_data=["Recommendation"],
    title="Priority Score vs Churn Probability"
)

st.plotly_chart(
    priority_chart,
    use_container_width=True
)

# =====================================================
# RECOMMENDATIONS
# =====================================================

st.subheader("Retention Recommendations")

recommendation_counts = (
    filtered_df["Recommendation"]
    .value_counts()
    .reset_index()
)

recommendation_counts.columns = [
    "Recommendation",
    "Count"
]

recommend_chart = px.bar(
    recommendation_counts,
    x="Count",
    y="Recommendation",
    orientation="h",
    title="Recommendation Distribution"
)

st.plotly_chart(
    recommend_chart,
    use_container_width=True
)

# =====================================================
# HIGH-RISK CUSTOMERS ONLY
# =====================================================

st.subheader("High-Risk Retention Queue")

high_risk_df = filtered_df[
    filtered_df["RiskLevel"] == "High"
]

st.dataframe(
    high_risk_df[
        [
            "ChurnProbability",
            "PriorityScore",
            "Recommendation"
        ]
    ],
    use_container_width=True
)

# =====================================================
# DOWNLOAD RESULTS
# =====================================================

st.download_button(
    label="Download Retention Plan",
    data=filtered_df.to_csv(index=False),
    file_name="retention_plan.csv",
    mime="text/csv"
)
`
