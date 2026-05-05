# pipeline.py
import logging
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from src.data_processing import load_data, clean_data, save_processed_data
from src.rfm_calculator import calculate_rfm, calculate_rfm_scores, segment_customers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/online_retail.csv"
PROCESSED_PATH = "data/processed/segmented_customers.csv"
N_CLUSTERS = 5
RANDOM_STATE = 42

CLUSTER_LABELS = {
    0: "🏆 Champions",
    1: "💎 Loyal Customers",
    2: "🌱 Potential Loyalists",
    3: "⚠️ At Risk",
    4: "💤 Hibernating",
}

def run_pipeline():
    # 1. Load & clean
    df_raw = load_data(RAW_PATH)
    df_clean = clean_data(df_raw)

    # 2. RFM
    rfm = calculate_rfm(df_clean)
    rfm = calculate_rfm_scores(rfm)
    rfm = segment_customers(rfm)

    # 3. Scale (Yeo-Johnson handles zeros and skew)
    pt = PowerTransformer(method='yeo-johnson')
    scaled = pt.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # 4. Cluster
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled)

    # 5. Human-readable label (assign by mean Monetary descending)
    cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
    label_map = {cluster: CLUSTER_LABELS[i] for i, cluster in enumerate(cluster_means.index)}
    rfm['Cluster_Label'] = rfm['Cluster'].map(label_map)

    # 6. Persist
    save_processed_data(rfm, PROCESSED_PATH)
    logger.info("Pipeline completed.")
    return rfm

if __name__ == "__main__":
    run_pipeline()