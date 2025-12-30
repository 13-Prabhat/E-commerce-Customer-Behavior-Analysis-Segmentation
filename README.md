# 📊 E-commerce Customer Behavior Analysis & Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

</div>

## 🎯 Project Overview

This project explores how customers interact with an online store by analyzing their purchase behavior and transaction patterns over time. It transforms raw e-commerce data into actionable business insights through data cleaning, exploratory analysis, RFM (Recency, Frequency, Monetary) segmentation, and machine learning-based clustering.

The goal is to help e-commerce businesses understand their customers better, identify valuable segments, and make data-driven decisions to improve marketing strategies, boost retention, and maximize revenue.

---

## 🚀 Key Objectives

- 📈 **Understand Customer Patterns**: Analyze buying behavior, order frequency, and spending habits
- 🎯 **Segment Customers**: Group customers into meaningful segments using RFM analysis and clustering
- 💡 **Generate Insights**: Identify high-value customers, at-risk customers, and new customer opportunities
- 📊 **Actionable Recommendations**: Provide marketing and retention strategies for each customer segment
- 🔮 **Predictive Analytics**: Build models to predict customer lifetime value and churn risk

---

## 🛠️ Tech Stack

**Language & Environment:**
- Python 3.8+
- Jupyter Notebook / JupyterLab

**Core Libraries:**
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`
- **Statistical Analysis**: `scipy`

---

## 📁 Project Structure

```
E-commerce-Customer-Behavior-Analysis-Segmentation/
│
├── data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda_customer_behavior.ipynb
│   ├── 03_rfm_feature_engineering.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_insights_recommendations.ipynb
│
├── src/
│   ├── data_processing.py      # Data cleaning utilities
│   ├── rfm_calculator.py       # RFM metrics calculation
│   └── visualization.py        # Custom plotting functions
│
├── reports/
│   ├── figures/                # Generated plots and charts
│   └── summary.md              # Final analysis summary
│
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 📊 Workflow & Methodology

### 1️⃣ **Data Collection & Cleaning**
- Load raw e-commerce transaction data
- Handle missing values, duplicates, and data type conversions
- Validate data integrity and consistency

### 2️⃣ **Exploratory Data Analysis (EDA)**
- Analyze customer demographics and purchase patterns
- Visualize distributions of orders, revenue, and products
- Identify trends and anomalies in the data

### 3️⃣ **Feature Engineering**
- Create customer-level aggregated metrics
- Calculate **RFM scores**:
  - **Recency**: Days since last purchase
  - **Frequency**: Total number of orders
  - **Monetary**: Total amount spent

### 4️⃣ **Customer Segmentation**
- Apply K-Means clustering on RFM features
- Identify optimal number of clusters using Elbow Method and Silhouette Score
- Label segments with business-friendly names

### 5️⃣ **Insights & Recommendations**
- Analyze characteristics of each segment
- Provide targeted marketing strategies
- Calculate segment revenue contribution

---

## 🎯 Expected Customer Segments

| Segment | Description | Marketing Strategy |
|---------|-------------|-------------------|
| 🏆 **Champions** | High RFM scores - best customers | VIP rewards, early access, personalized offers |
| 💎 **Loyal Customers** | High frequency, regular purchases | Loyalty programs, referral incentives |
| 🌱 **Potential Loyalists** | Recent buyers with potential | Engagement campaigns, product recommendations |
| ⚠️ **At Risk** | Previously active, now declining | Win-back campaigns, special discounts |
| 🆕 **New Customers** | Recent first-time buyers | Onboarding emails, welcome offers |
| 💤 **Hibernating** | Low recency and frequency | Re-engagement campaigns, surveys |

---

## 📈 Key Results

> 🔄 **In Progress** - Results will be updated as the project progresses through each phase.

**Preliminary Findings:**
- Total customers analyzed: _TBD_
- Number of segments identified: _TBD_
- Revenue contribution by segment: _TBD_
- Retention rate insights: _TBD_

---

## 🚀 How to Run This Project

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Install pip if not already installed
pip --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/13-Prabhat/E-commerce-Customer-Behavior-Analysis-Segmentation.git
cd E-commerce-Customer-Behavior-Analysis-Segmentation
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Run notebooks in sequence**
- Start with `01_data_loading_cleaning.ipynb`
- Follow the numbered order for best results

---

## 📊 Dataset

This project uses the **Online Retail Dataset** containing transaction data from a UK-based online retailer.

**Source**: [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

**Features:**
- InvoiceNo: Transaction ID
- StockCode: Product code
- Description: Product description
- Quantity: Number of items
- InvoiceDate: Transaction date and time
- UnitPrice: Price per unit
- CustomerID: Unique customer identifier
- Country: Customer location

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Prabhat**
- GitHub: [@13-Prabhat](https://github.com/13-Prabhat)
- Project Link: [E-commerce Customer Behavior Analysis](https://github.com/13-Prabhat/E-commerce-Customer-Behavior-Analysis-Segmentation)

---



## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Data cleaning and preprocessing techniques
- ✅ Exploratory data analysis and visualization
- ✅ Feature engineering for customer analytics
- ✅ Unsupervised machine learning (clustering)
- ✅ Business intelligence and actionable insights
- ✅ Professional documentation and version control

---

<div align="center">

### ⭐ If you find this project helpful, please give it a star!

**Made with ❤️ by Prabhat**

</div>
