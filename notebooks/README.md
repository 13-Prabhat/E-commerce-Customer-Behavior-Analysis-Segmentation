# 📔 Notebooks Directory

This directory contains Jupyter notebooks for the E-commerce Customer Behavior Analysis project.

## Notebook Execution Order

Execute the notebooks in the following sequence for best results:

### 1. `01_data_loading_cleaning.ipynb`
- Load raw e-commerce transaction data
- Handle missing values and duplicates
- Data type conversions and validation
- Output: Cleaned dataset saved to `data/processed/`

### 2. `02_eda_customer_behavior.ipynb`
- Exploratory Data Analysis (EDA)
- Customer purchase patterns
- Revenue and order distributions
- Key visualizations and insights

### 3. `03_rfm_feature_engineering.ipynb`
- Calculate RFM metrics per customer
- Recency: Days since last purchase
- Frequency: Total number of orders
- Monetary: Total amount spent
- Output: Customer-level RFM dataset

### 4. `04_customer_segmentation.ipynb`
- Apply K-Means clustering on RFM features
- Determine optimal number of clusters
- Label and interpret customer segments
- Segment profiling and visualization

### 5. `05_insights_recommendations.ipynb`
- Business insights per segment
- Marketing strategies and recommendations
- Revenue contribution analysis
- Final report generation

## Getting Started

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Run notebooks sequentially starting with `01_data_loading_cleaning.ipynb`

## Notes

- Each notebook is self-contained but depends on outputs from previous notebooks
- Intermediate results are saved in `data/processed/`
- Visualizations are exported to `reports/figures/`
