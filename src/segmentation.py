from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def perform_clustering(rfm_df, n_clusters=5, random_state=42):

    features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()

    transformer = PowerTransformer(method='yeo-johnson')

    scaled_features = transformer.fit_transform(features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    clusters = model.fit_predict(scaled_features)

    rfm_df['Cluster'] = clusters

    score = silhouette_score(
        scaled_features,
        clusters
    )

    return rfm_df, model, score