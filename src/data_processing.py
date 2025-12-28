"""Data Processing Utilities

This module provides utility functions for data cleaning and preprocessing
for the E-commerce Customer Behavior Analysis project.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath):
    """
    Load e-commerce transaction data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataframe
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    
    logger.info(f"Loading data from {filepath}...")
    
    # Try different encodings
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Could not read file with standard encodings: {e}")
    
    # Validate required columns
    required_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                       'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Clean and preprocess the transaction data.
    
    Args:
        df (pd.DataFrame): Raw dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    logger.info(f"Starting data cleaning... Initial shape: {df.shape}")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove rows with null CustomerID (B2B transactions)
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['CustomerID'].notna()]
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with null CustomerID")
    
    # Remove cancelled transactions (InvoiceNo starting with 'C')
    initial_rows = len(df_clean)
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    logger.info(f"Removed {initial_rows - len(df_clean)} cancelled transactions")
    
    # Remove rows with non-positive Quantity
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['Quantity'] > 0]
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with non-positive Quantity")
    
    # Remove rows with non-positive UnitPrice
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with non-positive UnitPrice")
    
    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
    
    # Remove rows with invalid dates
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['InvoiceDate'].notna()]
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with invalid dates")
    
    # Remove exact duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Convert CustomerID to integer
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    
    # Fill missing Description with 'Unknown'
    df_clean['Description'] = df_clean['Description'].fillna('Unknown')
    
    # Create TotalPrice column
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
    logger.info(f"Date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")
    logger.info(f"Unique customers: {df_clean['CustomerID'].nunique()}")
    
    return df_clean


def validate_data(df):
    """
    Validate cleaned data for quality checks.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    
    Returns:
        dict: Validation report
    """
    report = {
        'total_rows': len(df),
        'unique_customers': df['CustomerID'].nunique(),
        'unique_products': df['StockCode'].nunique(),
        'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
        'null_counts': df.isnull().sum().to_dict(),
        'negative_values': {
            'Quantity': (df['Quantity'] < 0).sum(),
            'UnitPrice': (df['UnitPrice'] < 0).sum(),
            'TotalPrice': (df['TotalPrice'] < 0).sum()
        },
        'total_revenue': df['TotalPrice'].sum()
    }
    
    logger.info("Data validation report:")
    for key, value in report.items():
        if key != 'null_counts':
            logger.info(f"  {key}: {value}")
    
    return report


def save_processed_data(df, output_path):
    """
    Save processed data to CSV file.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def get_data_summary(df):
    """
    Get summary statistics of the dataset.
    
    Args:
        df (pd.DataFrame): Dataframe to summarize
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'customers': {
            'total_unique': df['CustomerID'].nunique(),
            'avg_transactions': df.groupby('CustomerID')['InvoiceNo'].nunique().mean(),
            'avg_items_per_transaction': df.groupby('InvoiceNo')['Quantity'].sum().mean()
        },
        'revenue': {
            'total': df['TotalPrice'].sum(),
            'avg_per_transaction': df.groupby('InvoiceNo')['TotalPrice'].sum().mean(),
            'avg_per_customer': df.groupby('CustomerID')['TotalPrice'].sum().mean()
        },
        'products': {
            'total_unique': df['StockCode'].nunique(),
            'avg_price': df['UnitPrice'].mean(),
            'median_price': df['UnitPrice'].median()
        },
        'countries': df['Country'].nunique()
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    try:
        # Load data
        data_path = "data/raw/online_retail.csv"
        df = load_data(data_path)
        
        # Clean data
        df_clean = clean_data(df)
        
        # Validate data
        validation_report = validate_data(df_clean)
        
        # Get summary
        summary = get_data_summary(df_clean)
        print("\nData Summary:")
        print(summary)
        
        # Save processed data
        save_processed_data(df_clean, "data/processed/cleaned_data.csv")
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise
