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
                 price_col='UnitPrice', qty_col='Quantity',
                 snapshot_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Args:
        df (pd.DataFrame): Transaction dataframe
        customer_col (str): Column name for customer identifier
        date_col (str): Column name for transaction date
        price_col (str): Column name for unit price
        qty_col (str): Column name for quantity
        snapshot_date (datetime): Reference date for recency calculation.
                                 Defaults to max date + 1 day
    
    Returns:
        pd.DataFrame: RFM metrics for each customer with columns:
                     CustomerID, Recency, Frequency, Monetary
    
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger.info("Starting RFM calculation...")
    
    # Validate input dataframe
    required_cols = [customer_col, date_col, price_col, qty_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create working copy
    df_rfm = df.copy()
    
    # Ensure date column is datetime
    df_rfm[date_col] = pd.to_datetime(df_rfm[date_col])
    
    # Remove rows with null customer IDs
    df_rfm = df_rfm[df_rfm[customer_col].notna()]
    
    # Remove rows with non-positive values
    df_rfm = df_rfm[(df_rfm[qty_col] > 0) & (df_rfm[price_col] > 0)]
    
    # Calculate total price
    df_rfm['TotalPrice'] = df_rfm[qty_col] * df_rfm[price_col]
    
    # Set snapshot date (reference date for recency)
    if snapshot_date is None:
        snapshot_date = df_rfm[date_col].max() + timedelta(days=1)
    
    logger.info(f"Snapshot date for recency: {snapshot_date}")
    
    # Calculate RFM metrics
    rfm = df_rfm.groupby(customer_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
        'TotalPrice': ['count', 'sum']  # Frequency & Monetary
    })
    
    # Flatten column names
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    rfm.rename(columns={customer_col: 'CustomerID'}, inplace=True)
    
    # Ensure no negative values
    rfm['Recency'] = rfm['Recency'].clip(lower=0)
    rfm['Monetary'] = rfm['Monetary'].clip(lower=0)
    
    logger.info(f"RFM calculation completed for {len(rfm)} customers")
    logger.info(f"Recency range: {rfm['Recency'].min()}-{rfm['Recency'].max()} days")
    logger.info(f"Frequency range: {rfm['Frequency'].min()}-{rfm['Frequency'].max()} transactions")
    logger.info(f"Monetary range: ${rfm['Monetary'].min():.2f}-${rfm['Monetary'].max():.2f}")
    
    return rfm


def calculate_rfm_scores(rfm_df, method='quantile', n_bins=5):
    """
    Calculate RFM scores (1-5 scale) for each metric.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with Recency, Frequency, Monetary columns
        method (str): Scoring method - 'quantile' or 'equal'
        n_bins (int): Number of bins/scores (default: 5)
    
    Returns:
        pd.DataFrame: RFM dataframe with added R_Score, F_Score, M_Score columns
    """
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
    else:
        # Equal interval binning
        rfm_scored['R_Score'] = pd.cut(rfm_scored['Recency'], bins=n_bins,
                                        labels=range(n_bins, 0, -1))
        rfm_scored['F_Score'] = pd.cut(rfm_scored['Frequency'], bins=n_bins,
                                        labels=range(1, n_bins + 1))
        rfm_scored['M_Score'] = pd.cut(rfm_scored['Monetary'], bins=n_bins,
                                        labels=range(1, n_bins + 1))
    
    # Convert to integer
    rfm_scored['R_Score'] = rfm_scored['R_Score'].astype(int)
    rfm_scored['F_Score'] = rfm_scored['F_Score'].astype(int)
    rfm_scored['M_Score'] = rfm_scored['M_Score'].astype(int)
    
    # Calculate RFM combined score
    rfm_scored['RFM_Score'] = (rfm_scored['R_Score'].astype(str) +
                               rfm_scored['F_Score'].astype(str) +
                               rfm_scored['M_Score'].astype(str))
    
    logger.info("RFM scores calculated successfully")
    
    return rfm_scored


def segment_customers(rfm_df):
    """
    Segment customers based on RFM scores into meaningful business categories.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with R_Score, F_Score, M_Score columns
    
    Returns:
        pd.DataFrame: RFM dataframe with added 'Segment' column
    """
    rfm_segmented = rfm_df.copy()
    
    def assign_segment(row):
        """Assign customer segment based on RFM scores."""
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Champions: Best customers - high value, frequent, recent
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal Customers: Regular high-value customers
        elif r >= 3 and f >= 4 and m >= 3:
            return 'Loyal Customers'
        
        # Potential Loyalists: Recent customers with good frequency
        elif r >= 4 and f >= 2 and f <= 3 and m >= 2:
            return 'Potential Loyalists'
        
        # New Customers: Recent first-time or low-frequency buyers
        elif r >= 4 and f <= 2 and m >= 1:
            return 'New Customers'
        
        # At Risk: Were good customers but haven't purchased recently
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        
        # Can't Lose Them: High value but long time since purchase
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        
        # Hibernating: Low recency, low frequency, but some value
        elif r <= 2 and f <= 2 and m >= 2:
            return 'Hibernating'
        
        # Lost: Lowest scores, haven't engaged recently
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Lost'
        
        # About to Sleep: Below average recency and frequency
        elif r <= 3 and f <= 3 and m <= 3:
            return 'About to Sleep'
        
        # Need Attention: Above average but declining
        elif r == 3 and f >= 2 and m >= 2:
            return 'Need Attention'
        
        # Promising: Recent with moderate frequency and value
        elif r >= 3 and f >= 1 and m >= 1:
            return 'Promising'
        
        else:
            return 'Others'
    
    rfm_segmented['Segment'] = rfm_segmented.apply(assign_segment, axis=1)
    
    # Log segment distribution
    segment_counts = rfm_segmented['Segment'].value_counts()
    logger.info("Customer segmentation completed:")
    for segment, count in segment_counts.items():
        logger.info(f"  {segment}: {count} customers ({count/len(rfm_segmented)*100:.1f}%)")
    
    return rfm_segmented


def get_rfm_summary(rfm_df):
    """
    Generate summary statistics for RFM analysis.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_customers': len(rfm_df),
        'rfm_metrics': {
            'recency': {
                'mean': rfm_df['Recency'].mean(),
                'median': rfm_df['Recency'].median(),
                'min': rfm_df['Recency'].min(),
                'max': rfm_df['Recency'].max()
            },
            'frequency': {
                'mean': rfm_df['Frequency'].mean(),
                'median': rfm_df['Frequency'].median(),
                'min': rfm_df['Frequency'].min(),
                'max': rfm_df['Frequency'].max()
            },
            'monetary': {
                'mean': rfm_df['Monetary'].mean(),
                'median': rfm_df['Monetary'].median(),
                'min': rfm_df['Monetary'].min(),
                'max': rfm_df['Monetary'].max(),
                'total': rfm_df['Monetary'].sum()
            }
        }
    }
    
    # Add segment statistics if available
    if 'Segment' in rfm_df.columns:
        summary['segments'] = rfm_df['Segment'].value_counts().to_dict()
        summary['segment_revenue'] = rfm_df.groupby('Segment')['Monetary'].sum().to_dict()
    
    return summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        # This would typically use cleaned data from data_processing.py
        print("RFM Calculator module loaded successfully")
        print("Use calculate_rfm() to compute RFM metrics from transaction data")
    except Exception as e:
        logger.error(f"Error: {e}")
