"""
RFM Metrics Calculator

This module calculates RFM (Recency, Frequency, Monetary) metrics
for customer segmentation and behavioral analysis.
"""

import pandas as pd
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_rfm(
    df,
    customer_col='CustomerID',
    date_col='InvoiceDate',
    invoice_col='InvoiceNo',
    price_col='TotalPrice',
    snapshot_date=None
):
    """
    Calculate RFM metrics for each customer.

    Args:
        df (pd.DataFrame): Transaction dataframe
        customer_col (str): Customer identifier column
        date_col (str): Transaction date column
        invoice_col (str): Invoice/order identifier column
        price_col (str): Monetary value column
        snapshot_date (datetime): Reference date for recency

    Returns:
        pd.DataFrame: RFM dataframe
    """

    logger.info("Starting RFM calculation...")

    # Validate input dataframe
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    required_cols = [
        customer_col,
        date_col,
        invoice_col,
        price_col
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # Create copy
    df_rfm = df.copy()

    # Remove missing customers
    df_rfm = df_rfm.dropna(subset=[customer_col])

    # Convert dates safely
    df_rfm[date_col] = pd.to_datetime(
        df_rfm[date_col],
        errors='coerce'
    )

    # Remove invalid dates
    df_rfm = df_rfm.dropna(subset=[date_col])

    # Remove negative/zero monetary transactions
    df_rfm = df_rfm[df_rfm[price_col] > 0]

    if df_rfm.empty:
        raise ValueError(
            "No valid transactions remain after cleaning."
        )

    # Snapshot date
    if snapshot_date is None:
        snapshot_date = (
            df_rfm[date_col].max() + timedelta(days=1)
        )

    logger.info(
        f"Snapshot date for recency: {snapshot_date}"
    )

    # RFM Calculation
    rfm = (
        df_rfm.groupby(customer_col)
        .agg(
            Recency=(
                date_col,
                lambda x: (snapshot_date - x.max()).days
            ),
            Frequency=(invoice_col, 'nunique'),
            Monetary=(price_col, 'sum')
        )
        .reset_index()
    )

    # Safety clipping
    rfm['Recency'] = rfm['Recency'].clip(lower=0)
    rfm['Monetary'] = rfm['Monetary'].clip(lower=0)

    logger.info(
        f"RFM calculation completed for {len(rfm)} customers"
    )

    logger.info(
        f"Monetary range: £{rfm['Monetary'].min():.2f} - "
        f"£{rfm['Monetary'].max():.2f}"
    )

    return rfm


def calculate_rfm_scores(
    rfm_df,
    method='quantile',
    n_bins=5
):
    """
    Calculate RFM scores on a 1–5 scale.
    """

    if rfm_df.empty:
        raise ValueError("RFM dataframe is empty.")

    rfm_scored = rfm_df.copy()

    if method == 'quantile':

        # Use ranking first to avoid qcut duplicate edge issues
        recency_rank = rfm_scored['Recency'].rank(
            method='first'
        )

        frequency_rank = rfm_scored['Frequency'].rank(
            method='first'
        )

        monetary_rank = rfm_scored['Monetary'].rank(
            method='first'
        )

        rfm_scored['R_Score'] = pd.qcut(
            recency_rank,
            q=n_bins,
            labels=range(n_bins, 0, -1)
        )

        rfm_scored['F_Score'] = pd.qcut(
            frequency_rank,
            q=n_bins,
            labels=range(1, n_bins + 1)
        )

        rfm_scored['M_Score'] = pd.qcut(
            monetary_rank,
            q=n_bins,
            labels=range(1, n_bins + 1)
        )

    else:
        raise ValueError(
            f"Unsupported scoring method: {method}"
        )

    rfm_scored['R_Score'] = (
        rfm_scored['R_Score'].astype(int)
    )

    rfm_scored['F_Score'] = (
        rfm_scored['F_Score'].astype(int)
    )

    rfm_scored['M_Score'] = (
        rfm_scored['M_Score'].astype(int)
    )

    return rfm_scored


def segment_customers(rfm_df):
    """
    Segment customers based on RFM scores.
    """

    if rfm_df.empty:
        raise ValueError("Input dataframe is empty.")

    required_cols = [
        'R_Score',
        'F_Score',
        'M_Score'
    ]

    missing_cols = [
        col for col in required_cols
        if col not in rfm_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    rfm_segmented = rfm_df.copy()

    def assign_segment(row):

        r = row['R_Score']
        f = row['F_Score']
        m = row['M_Score']

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"

        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"

        elif r >= 4 and f <= 2:
            return "New Customers"

        elif r >= 3 and f >= 2 and m >= 2:
            return "Potential Loyalists"

        elif r <= 2 and f >= 3:
            return "At Risk"

        elif r <= 2 and f <= 2 and m >= 3:
            return "Hibernating"

        elif r == 3:
            return "Need Attention"

        else:
            return "Lost"

    rfm_segmented['Segment'] = (
        rfm_segmented.apply(assign_segment, axis=1)
    )

    logger.info(
        "Customer segmentation completed successfully."
    )

    return rfm_segmented