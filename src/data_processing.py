"""Data Processing Utilities

This module provides utility functions for data cleaning and preprocessing
for the E-commerce Customer Behavior Analysis project.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_data(filepath):
    """
    Load e-commerce transaction data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Remove rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # Remove rows with missing Description
    df = df.dropna(subset=['Description'])
    
    return df


def remove_duplicates(df):
    """
    Remove duplicate records from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    return df.drop_duplicates()


def filter_valid_transactions(df):
    """
    Filter out invalid transactions (negative quantities, prices).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Keep only positive quantities and prices
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    return df


def add_total_amount(df):
    """
    Add TotalAmount column (Quantity * UnitPrice).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with TotalAmount column
    """
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df


def convert_date_format(df, date_column='InvoiceDate'):
    """
    Convert date column to datetime format.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: Dataframe with converted dates
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return df


def clean_data(df):
    """
    Apply all cleaning steps to the dataframe.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print(f"Initial records: {len(df)}")
    
    df = handle_missing_values(df)
    print(f"After removing missing values: {len(df)}")
    
    df = remove_duplicates(df)
    print(f"After removing duplicates: {len(df)}")
    
    df = filter_valid_transactions(df)
    print(f"After filtering valid transactions: {len(df)}")
    
    df = convert_date_format(df)
    df = add_total_amount(df)
    
    print(f"Final cleaned records: {len(df)}")
    
    return df


def save_cleaned_data(df, filepath):
    """
    Save cleaned data to CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        filepath (str): Output file path
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
