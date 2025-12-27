"""Visualization Utilities

This module provides custom plotting functions for data visualization
in the E-commerce Customer Behavior Analysis project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_plot_style():
    """Set default plot style for consistent visualization."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_distribution(data, column, title=None, xlabel=None, bins=30):
    """
    Plot distribution histogram.
    
    Args:
        data (pd.DataFrame or pd.Series): Input data
        column (str): Column name to plot (if DataFrame)
        title (str): Plot title
        xlabel (str): X-axis label
        bins (int): Number of bins
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(data, pd.DataFrame):
        values = data[column]
    else:
        values = data
    
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel or column)
    plt.ylabel('Frequency')
    plt.title(title or f'Distribution of {column}')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_rfm_distribution(rfm_df):
    """
    Plot distributions of RFM metrics.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with RFM metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Recency
    axes[0].hist(rfm_df['Recency'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Recency (days)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Recency')
    axes[0].grid(True, alpha=0.3)
    
    # Frequency
    axes[1].hist(rfm_df['Frequency'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1].set_xlabel('Frequency (orders)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Monetary
    axes[2].hist(rfm_df['Monetary'], bins=30, edgecolor='black', alpha=0.7, color='salmon')
    axes[2].set_xlabel('Monetary (£)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Monetary')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_segment_distribution(rfm_df):
    """
    Plot customer segment distribution.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with Customer_Segment column
    """
    plt.figure(figsize=(12, 6))
    
    segment_counts = rfm_df['Customer_Segment'].value_counts()
    
    plt.bar(segment_counts.index, segment_counts.values, edgecolor='black', alpha=0.7)
    plt.xlabel('Customer Segment')
    plt.ylabel('Number of Customers')
    plt.title('Customer Segment Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_segment_revenue(rfm_df):
    """
    Plot revenue contribution by customer segment.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with Customer_Segment and Monetary columns
    """
    plt.figure(figsize=(12, 6))
    
    segment_revenue = rfm_df.groupby('Customer_Segment')['Monetary'].sum().sort_values(ascending=False)
    
    plt.bar(segment_revenue.index, segment_revenue.values, edgecolor='black', alpha=0.7, color='gold')
    plt.xlabel('Customer Segment')
    plt.ylabel('Total Revenue (£)')
    plt.title('Revenue Contribution by Customer Segment')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_scatter_rfm(rfm_df, x_metric='Frequency', y_metric='Monetary', 
                     hue='Customer_Segment'):
    """
    Plot scatter plot of RFM metrics.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with RFM metrics
        x_metric (str): X-axis metric
        y_metric (str): Y-axis metric
        hue (str): Color grouping column
    """
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(data=rfm_df, x=x_metric, y=y_metric, hue=hue, 
                    s=100, alpha=0.6, palette='Set2')
    
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f'{x_metric} vs {y_metric} by {hue}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_heatmap(data, title='Correlation Heatmap'):
    """
    Plot correlation heatmap.
    
    Args:
        data (pd.DataFrame): DataFrame with numeric columns
        title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    
    correlation = data.corr()
    
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_segment_summary_table(summary_df):
    """
    Display segment summary as a formatted table plot.
    
    Args:
        summary_df (pd.DataFrame): Segment summary DataFrame
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['lightblue'] * len(summary_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title('Customer Segment Summary Statistics', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
