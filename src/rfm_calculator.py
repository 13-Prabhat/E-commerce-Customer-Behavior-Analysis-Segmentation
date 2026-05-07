"""RFM Metrics Calculator

This module calculates RFM (Recency, Frequency, Monetary) metrics
for customer segmentation and behavioral analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_rfm(df, customer_col='CustomerID', date_col='InvoiceDate',
                 invoice_col='InvoiceNo', price_col='TotalPrice',
                 snapshot_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Args:
        df (pd.DataFrame): Transaction dataframe
        customer_col (str): Column name for customer identifier
        date_col (str): Column name for transaction date
        invoice_col (str): Column name for distinct invoice/order identifier
        price_col (str): Column name for total price per line item
        snapshot_date (datetime): Reference date for recency calculation.
                                  Defaults to max date + 1 day
    
    Returns:
        pd.DataFrame: RFM metrics for each customer
    """
    logger.info("Starting RFM calculation...")
    
    # Ensure date column is datetime
    df_rfm = df.copy()
    df_rfm[date_col] = pd.to_datetime(df_rfm[date_col])
    
    # Set snapshot date (reference date for recency)
   if snapshot_date is None:
    snapshot_date = df[date_col].max() + timedelta(days=1)
    
    logger.info(f"Snapshot date for recency: {snapshot_date}")
    
    # Calculate RFM metrics
    # Fixed: Frequency now counts UNIQUE invoices, not raw rows[cite: 1]
    rfm = df_rfm.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(invoice_col, 'nunique'),   # Corrected logic[cite: 1]
        Monetary=(price_col, 'sum')
    ).reset_index()
    
    # Ensure no negative values
    rfm['Recency'] = rfm['Recency'].clip(lower=0)
    rfm['Monetary'] = rfm['Monetary'].clip(lower=0)
    
    logger.info(f"RFM calculation completed for {len(rfm)} customers")
    # Fixed: Changed currency label to £ per audit findings[cite: 1]
    logger.info(f"Monetary range: £{rfm['Monetary'].min():.2f} - £{rfm['Monetary'].max():.2f}")
    
    return rfm


def calculate_rfm_scores(rfm_df, method='quantile', n_bins=5):
    """Calculate RFM scores (1-5 scale) for each metric."""
    rfm_scored = rfm_df.copy()
    
    if method == 'quantile':
        # Recency: lower is better, so reverse scoring
        rfm_scored['R_Score'] = pd.qcut(rfm_scored['Recency'], q=n_bins,
                                         labels=range(n_bins, 0, -1), duplicates='drop')
        # Frequency & Monetary: higher is better
        rfm_scored['F_Score'] = pd.qcut(rfm_scored['Frequency'], q=n_bins,
                                         labels=range(1, n_bins + 1), duplicates='drop')
        rfm_scored['M_Score'] = pd.qcut(rfm_scored['Monetary'], q=n_bins,
                                         labels=range(1, n_bins + 1), duplicates='drop')
    
    # Convert to integer
    rfm_scored['R_Score'] = rfm_scored['R_Score'].astype(int)
    rfm_scored['F_Score'] = rfm_scored['F_Score'].astype(int)
    rfm_scored['M_Score'] = rfm_scored['M_Score'].astype(int)
    
    return rfm_scored


def segment_customers(rfm_df):
    """Segment customers based on RFM scores into business categories."""
    rfm_segmented = rfm_df.copy()
    
    def assign_segment(row):
        """Assign customer segment based on corrected priority logic[cite: 1]."""
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Fixed order: High priority and specific rules first[cite: 1]
        if   r >= 4 and f >= 4 and m >= 4:  return "Champions"
        elif r <= 2 and f >= 4 and m >= 4:  return "Can't Lose Them" # Now reachable[cite: 1]
        elif r >= 3 and f >= 3 and m >= 3:  return "Loyal Customers"
        elif r >= 4 and f <= 2:            return "New Customers"
        elif r >= 3 and f >= 2:            return "Potential Loyalists"
        elif r <= 2 and f >= 3:            return "At Risk"
        elif r <= 2 and f <= 2 and m >= 3: return "Hibernating"
        elif r == 3:                         return "Need Attention"
        else:                                return "Lost"
    
    rfm_segmented['Segment'] = rfm_segmented.apply(assign_segment, axis=1)
    return rfm_segmented