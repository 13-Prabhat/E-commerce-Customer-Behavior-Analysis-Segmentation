"""RFM Metrics Calculator

This module calculates RFM (Recency, Frequency, Monetary) metrics
for customer segmentation analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_rfm(df, customer_id='CustomerID', invoice_date='InvoiceDate', 
                  total_amount='TotalAmount', reference_date=None):
    """
    Calculate RFM metrics for each customer.
    
    Args:
        df (pd.DataFrame): Transaction dataframe
        customer_id (str): Column name for customer ID
        invoice_date (str): Column name for invoice date
        total_amount (str): Column name for total amount
        reference_date (datetime): Reference date for recency calculation
        
    Returns:
        pd.DataFrame: DataFrame with RFM metrics per customer
    """
    if reference_date is None:
        reference_date = df[invoice_date].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby(customer_id).agg({
        invoice_date: lambda x: (reference_date - x.max()).days,  # Recency
        'InvoiceNo': 'count',  # Frequency
        total_amount: 'sum'  # Monetary
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    
    return rfm


def score_rfm(rfm_df, recency_col='Recency', frequency_col='Frequency', 
              monetary_col='Monetary', quantiles=5):
    """
    Assign RFM scores using quantile-based scoring.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with RFM metrics
        recency_col (str): Column name for recency
        frequency_col (str): Column name for frequency
        monetary_col (str): Column name for monetary
        quantiles (int): Number of quantiles for scoring (default 5)
        
    Returns:
        pd.DataFrame: DataFrame with RFM scores added
    """
    # Create a copy to avoid modifying original
    rfm = rfm_df.copy()
    
    # Score Recency (lower is better, so reverse)
    rfm['R_Score'] = pd.qcut(rfm[recency_col], q=quantiles, labels=range(quantiles, 0, -1), duplicates='drop')
    
    # Score Frequency (higher is better)
    rfm['F_Score'] = pd.qcut(rfm[frequency_col], q=quantiles, labels=range(1, quantiles+1), duplicates='drop')
    
    # Score Monetary (higher is better)
    rfm['M_Score'] = pd.qcut(rfm[monetary_col], q=quantiles, labels=range(1, quantiles+1), duplicates='drop')
    
    # Convert to integer
    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)
    
    # Create RFM Score (combination)
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Create RFM Segment Score (simple sum)
    rfm['RFM_Segment_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    return rfm


def segment_customers(rfm_df):
    """
    Segment customers into business-friendly groups based on RFM scores.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with RFM scores
        
    Returns:
        pd.DataFrame: DataFrame with customer segments
    """
    rfm = rfm_df.copy()
    
    def assign_segment(row):
        r_score = row['R_Score']
        f_score = row['F_Score']
        m_score = row['M_Score']
        
        # Champions - High R, F, M
        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            return 'Champions'
        
        # Loyal Customers - High F, M
        elif f_score >= 4 and m_score >= 4:
            return 'Loyal Customers'
        
        # Potential Loyalists - Recent and moderate frequency
        elif r_score >= 4 and f_score >= 3:
            return 'Potential Loyalists'
        
        # At Risk - Low recency but was good customer
        elif r_score <= 2 and f_score >= 3 and m_score >= 3:
            return 'At Risk'
        
        # New Customers - Recent but low frequency
        elif r_score >= 4 and f_score <= 2:
            return 'New Customers'
        
        # Promising - Recent with potential
        elif r_score >= 3 and f_score >= 2:
            return 'Promising'
        
        # Need Attention - Moderate scores
        elif r_score >= 2 and f_score >= 2:
            return 'Need Attention'
        
        # Hibernating - Low recency and frequency
        elif r_score <= 2 and f_score <= 2:
            return 'Hibernating'
        
        else:
            return 'Other'
    
    rfm['Customer_Segment'] = rfm.apply(assign_segment, axis=1)
    
    return rfm


def get_segment_summary(rfm_df):
    """
    Generate summary statistics for each customer segment.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with customer segments
        
    Returns:
        pd.DataFrame: Summary statistics by segment
    """
    summary = rfm_df.groupby('Customer_Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum']
    }).round(2)
    
    summary.columns = ['Customer_Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue']
    summary = summary.reset_index()
    
    # Calculate percentage
    summary['Percentage'] = (summary['Customer_Count'] / summary['Customer_Count'].sum() * 100).round(2)
    
    return summary.sort_values('Total_Revenue', ascending=False)
