import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# Theme Toggle
theme = st.sidebar.radio("🎨 Theme Mode", ["Light", "Dark"])

# Title
st.title("🧠 Customer Segmentation Dashboard")
st.caption("Interactive K-Means Clustering & Customer Behavior Analysis")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# Sidebar Controls
st.sidebar.header("⚙️ Clustering Controls")

feature_map = {
    "Age": "Age",
    "Annual Income": "Annual Income (k$)",
    "Spending Score": "Spending Score (1-100)"
}

selected_features = st.sidebar.multiselect(
    "Select features for clustering",
    list(feature_map.keys()),
    default=["Annual Income", "Spending Score"]
)

if len(selected_features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 5)

# Prepare Features
X = df[[feature_map[f] for f in selected_features]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Model
kmeans = KMeans(
    n_clusters=k,
    init="k-means++",
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", len(df))
c2.metric("Features Used", len(selected_features))
c3.metric("Clusters", k)

# Visualization
st.subheader("📈 Cluster Visualization")

if len(selected_features) == 2:
    fig = px.scatter(
        df,
        x=feature_map[selected_features[0]],
        y=feature_map[selected_features[1]],
        color="Cluster",
        template="plotly_dark" if theme == "Dark" else "plotly_white"
    )
else:
    fig = px.scatter_3d(
        df,
        x=feature_map[selected_features[0]],
        y=feature_map[selected_features[1]],
        z=feature_map[selected_features[2]],
        color="Cluster",
        template="plotly_dark" if theme == "Dark" else "plotly_white"
    )

st.plotly_chart(fig, width="stretch")

# Cluster Distribution
fig_bar = px.histogram(
    df,
    x="Cluster",
    color="Cluster",
    title="Customers per Cluster",
    template="plotly_dark" if theme == "Dark" else "plotly_white"
)

st.plotly_chart(fig_bar, width="stretch")

# Summary Table
st.subheader("📌 Cluster Summary")

summary = (
    df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    .mean()
    .round(2)
)

st.dataframe(summary, width="stretch")

# Download
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Segmented Data",
    csv,
    "segmented_customers.csv",
    "text/csv"
)

# Insights
st.subheader("💡 Business Insights")

st.markdown("""
- **High income & high spending** → Premium customers  
- **High income & low spending** → Potential customers  
- **Low income & high spending** → Impulse buyers  
- **Low income & low spending** → Budget customers  
- **Moderate behavior** → Regular customers  
""")

# Footer
st.markdown("---")
st.markdown("👤 **P. Vaishnavi** | Aspiring Data Analyst | Machine Learning Enthusiast")
