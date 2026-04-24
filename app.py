import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="E-commerce Customer Segmenter",
    page_icon="📊",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .medium-font { font-size:16px !important; }
    .reportview-container .main .block-container{ padding-top: 2rem; }
    div[data-testid="stMetricValue"] > div { font-size: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# --- TITLE & OVERVIEW ---
st.title("📊 E-commerce Customer Behavior Analysis & Segmentation")
st.markdown("## Interactive BI Dashboard")
st.markdown("""
Transforming raw transaction data into actionable business intelligence using 
**RFM (Recency, Frequency, Monetary)** analysis and **K-Means Clustering**.
""")
st.divider()

# --- FILE PATH ---
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'segmented_customers.csv')

# --- DATA LOADING ---
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"❌ Could not find data at `{path}`. Run preprocessing notebooks first.")
        st.stop()
    df = pd.read_csv(path)

    # Type safety
    if 'CustomerID' in df.columns:
        df['CustomerID'] = df['CustomerID'].astype(str)

    return df

df = load_data(PROCESSED_DATA_PATH)

# --- GLOBAL KPIs ---
col1, col2, col3, col4 = st.columns(4)

total_customers = df['CustomerID'].nunique()
total_revenue = df['Monetary'].sum()
avg_freq = df['Frequency'].mean()
avg_recency = df['Recency'].mean()

with col1:
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    st.metric("Total Revenue (GBP)", f"£{total_revenue:,.2f}")

with col3:
    st.metric("Avg Order Frequency", f"{avg_freq:.2f}")

with col4:
    st.metric("Avg Recency (Days)", f"{avg_recency:.1f}")

st.divider()

# --- SIDEBAR FILTER ---
st.sidebar.header("Filter & Deep Dive")

segments = sorted(df['Cluster_Label'].dropna().unique())

selected_segments = st.sidebar.multiselect(
    "Select Customer Segments:",
    options=segments,
    default=segments
)

# --- FILTERED DATA ---
filtered_df = df[df['Cluster_Label'].isin(selected_segments)]

# Handle empty selection
if filtered_df.empty:
    st.warning("⚠️ No data available for selected segments.")
    st.stop()

# --- DOWNLOAD BUTTON ---
st.sidebar.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_customers.csv",
    mime="text/csv"
)

# --- CHART LAYOUT ---
col_left, col_right = st.columns(2)

# --- CHART 1: REVENUE DISTRIBUTION ---
with col_left:
    st.markdown("### 💰 Revenue Contribution by Segment")

    revenue_by_segment = (
        df.groupby('Cluster_Label')['Monetary']
        .sum()
        .reset_index()
    )

    fig_donut = px.pie(
        revenue_by_segment,
        values='Monetary',
        names='Cluster_Label',
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(showlegend=False)

    st.plotly_chart(fig_donut, use_container_width=True)

# --- CHART 2: 3D RFM SCATTER ---
with col_right:
    st.markdown("### 📦 3D RFM Cluster View")

    fig_3d = px.scatter_3d(
        filtered_df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster_Label',
        opacity=0.7,
        hover_data=['CustomerID'],
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Recency (Days)',
            yaxis_title='Frequency (Orders)',
            zaxis_title='Monetary (Value)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# --- STRATEGY TABLE ---
st.markdown("## 🎯 Segment Strategies")

strategy_data = {
    'Segment': [
        '🏆 Champions', '💎 Loyal Customers', '🌱 Potential Loyalists',
        '🆕 New Customers', '⚠️ At Risk', '💤 Hibernating'
    ],
    'Key Characteristics': [
        'Bought recently, frequently, high spend.',
        'Regular buyers, respond to promotions.',
        'Recent buyers, moderate frequency.',
        'New but low frequency.',
        'Used to buy often, now inactive.',
        'Low activity overall.'
    ],
    'Strategy': [
        'VIP rewards, early access.',
        'Upsell, loyalty programs.',
        'Engage with recommendations.',
        'Onboarding & welcome offers.',
        'Win-back campaigns.',
        'Low-cost re-engagement.'
    ]
}

strategy_df = pd.DataFrame(strategy_data)
st.table(strategy_df)

# --- FOOTER ---
st.divider()
st.markdown("""
**Author:** Prabhat  
🔗 GitHub: https://github.com/13-Prabhat/E-commerce-Customer-Behavior-Analysis-Segmentation  
Built with Streamlit
""")
