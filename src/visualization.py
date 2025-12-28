"""Visualization Utilities

This module provides custom visualization functions for RFM analysis and customer behavior
in the E-commerce Customer Behavior Analysis project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Set default plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def set_plot_style():
    """Set default plot style for consistent visualizations."""
    sns.set_palette('husl')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'


def plot_rfm_distributions(rfm_df, save_path=None, show_plot=True):
    """
    Plot distributions of RFM metrics.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe with Recency, Frequency, Monetary
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recency distribution
    axes[0].hist(rfm_df['Recency'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Recency (days)')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Recency Distribution')
    axes[0].axvline(rfm_df['Recency'].mean(), color='red', linestyle='--', label=f'Mean: {rfm_df["Recency"].mean():.1f}')
    axes[0].legend()
    
    # Frequency distribution
    axes[1].hist(rfm_df['Frequency'], bins=50, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Frequency (transactions)')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_title('Frequency Distribution')
    axes[1].axvline(rfm_df['Frequency'].mean(), color='red', linestyle='--', label=f'Mean: {rfm_df["Frequency"].mean():.1f}')
    axes[1].legend()
    
    # Monetary distribution  
    axes[2].hist(rfm_df['Monetary'], bins=50, color='lightcoral', edgecolor='black')
    axes[2].set_xlabel('Monetary (revenue)')
    axes[2].set_ylabel('Number of Customers')
    axes[2].set_title('Monetary Distribution')
    axes[2].axvline(rfm_df['Monetary'].mean(), color='red', linestyle='--', label=f'Mean: ${rfm_df["Monetary"].mean():.2f}')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"RFM distributions plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rfm_scatter(rfm_df, x='Recency', y='Monetary', hue='Frequency',
                    save_path=None, show_plot=True):
    """
    Create scatter plot of RFM metrics.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        hue (str): Column name for color coding
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(rfm_df[x], rfm_df[y], c=rfm_df[hue], 
                         cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label=hue)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} vs {x} (colored by {hue})')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"RFM scatter plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_segment_distribution(rfm_df, save_path=None, show_plot=True):
    """
    Plot customer segment distribution.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe with 'Segment' column
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    if 'Segment' not in rfm_df.columns:
        logger.warning("No 'Segment' column found in dataframe")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    segment_counts = rfm_df['Segment'].value_counts()
    axes[0].barh(segment_counts.index, segment_counts.values, color='steelblue')
    axes[0].set_xlabel('Number of Customers')
    axes[0].set_title('Customer Count by Segment')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Percentage pie chart
    axes[1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
               startangle=90)
    axes[1].set_title('Customer Segment Distribution (%)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Segment distribution plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_segment_revenue(rfm_df, save_path=None, show_plot=True):
    """
    Plot revenue contribution by customer segment.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe with 'Segment' and 'Monetary' columns
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    if 'Segment' not in rfm_df.columns or 'Monetary' not in rfm_df.columns:
        logger.warning("Required columns not found in dataframe")
        return
    
    segment_revenue = rfm_df.groupby('Segment')['Monetary'].sum().sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(segment_revenue.index, segment_revenue.values, color='teal')
    plt.xlabel('Total Revenue ($)')
    plt.title('Revenue Contribution by Customer Segment')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Segment revenue plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rfm_heatmap(rfm_df, save_path=None, show_plot=True):
    """
    Plot correlation heatmap of RFM metrics.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    # Select only numeric RFM columns
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    available_cols = [col for col in rfm_cols if col in rfm_df.columns]
    
    if len(available_cols) < 2:
        logger.warning("Not enough RFM columns for correlation plot")
        return
    
    corr = rfm_df[available_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
               square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('RFM Metrics Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"RFM heatmap saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_segment_characteristics(rfm_df, save_path=None, show_plot=True):
    """
    Plot average RFM characteristics by segment.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe with 'Segment' column
        save_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    if 'Segment' not in rfm_df.columns:
        logger.warning("No 'Segment' column found in dataframe")
        return
    
    segment_stats = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Average Recency by segment
    segment_stats['Recency'].sort_values().plot(kind='barh', ax=axes[0], color='skyblue')
    axes[0].set_xlabel('Average Recency (days)')
    axes[0].set_title('Average Recency by Segment')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Average Frequency by segment
    segment_stats['Frequency'].sort_values().plot(kind='barh', ax=axes[1], color='lightgreen')
    axes[1].set_xlabel('Average Frequency')
    axes[1].set_title('Average Frequency by Segment')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Average Monetary by segment
    segment_stats['Monetary'].sort_values().plot(kind='barh', ax=axes[2], color='lightcoral')
    axes[2].set_xlabel('Average Monetary ($)')
    axes[2].set_title('Average Monetary by Segment')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Segment characteristics plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_rfm_report(rfm_df, output_dir='reports/figures'):
    """
    Generate all RFM visualization plots and save them.
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe
        output_dir (str): Directory to save plots
    """
    logger.info("Generating comprehensive RFM visualization report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_rfm_distributions(rfm_df, 
                          save_path=f'{output_dir}/rfm_distributions.png',
                          show_plot=False)
    
    plot_rfm_scatter(rfm_df, 
                    save_path=f'{output_dir}/rfm_scatter.png',
                    show_plot=False)
    
    plot_rfm_heatmap(rfm_df, 
                    save_path=f'{output_dir}/rfm_heatmap.png',
                    show_plot=False)
    
    if 'Segment' in rfm_df.columns:
        plot_segment_distribution(rfm_df, 
                                 save_path=f'{output_dir}/segment_distribution.png',
                                 show_plot=False)
        
        plot_segment_revenue(rfm_df, 
                           save_path=f'{output_dir}/segment_revenue.png',
                           show_plot=False)
        
        plot_segment_characteristics(rfm_df, 
                                    save_path=f'{output_dir}/segment_characteristics.png',
                                    show_plot=False)
    
    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Visualization module loaded successfully")
    print("Use functions like plot_rfm_distributions(), plot_segment_distribution() to create visualizations")
