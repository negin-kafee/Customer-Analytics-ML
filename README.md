# Customer Analytics ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-red.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end machine learning pipeline for customer analytics**: Spending prediction, campaign response classification, and customer segmentation using scikit-learn, XGBoost, and TensorFlow.

---

## 🎯 Key Results

| Task | Best Model | Performance | Business Impact |
|------|------------|-------------|-----------------|
| **Spending Prediction** | Random Forest | R² = **0.970** | Predict 97% of spending variance |
| **Campaign Response** | Random Forest | ROC-AUC = **0.875** | Identify 85%+ of responders |
| **Customer Segmentation** | K-Means (Optimized) | Silhouette = **0.614** | 4 actionable segments |
| **Deep Learning** | MLP | R² = 0.953 / AUC = 0.871 | Competitive with tree models |

### 📊 Model Performance Visualization

```
REGRESSION (R² Score)
Random Forest   ████████████████████████████████████████ 97.0%
XGBoost         ███████████████████████████████████████  96.9%
MLP Neural Net  █████████████████████████████████████    95.3%

CLASSIFICATION (ROC-AUC)
Random Forest   ████████████████████████████████████████ 87.5%
XGBoost         ███████████████████████████████████████  87.2%
MLP Neural Net  ███████████████████████████████████████  87.1%

CLUSTERING (Silhouette Score - After Optimization)
Optimized       █████████████████████████████████████    61.4%
Original        ████████                                 16.5%
                ↑ 272% improvement through feature engineering
```

---

## 📋 Project Overview

This project demonstrates a **complete machine learning workflow** for customer analytics, from exploratory data analysis to production-ready models. It showcases:

- **15+ ML algorithms** compared and evaluated
- **Custom sklearn transformers** for reproducible preprocessing
- **Hyperparameter optimization** with cross-validation
- **Imbalanced classification** handling (15% positive class)
- **Clustering optimization** achieving 272% improvement
- **Deep learning** with TensorFlow/Keras
- **Cold Start handling** for new customer predictions

### Business Questions Answered

1. 💰 **How much will a customer spend?** → Regression models predict with 97% accuracy (existing) / 78% (new customers)
2. 📧 **Will they respond to campaigns?** → Classification identifies 85% of responders
3. 👥 **What customer segments exist?** → 4 distinct, actionable segments identified
4. 🧠 **Can deep learning help?** → Competitive but tree models win on small data
5. 🆕 **What about NEW customers?** → Demographics-only model achieves R² = 0.78

## 📁 Project Structure

```
customer-analytics-ml-pipeline/
├── 📓 Notebooks
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_regression.ipynb             # Spending Prediction (R²=0.97) - Existing Customers
│   ├── 02b_regression_new_customers.ipynb  # Cold Start Model (R²=0.78) - New Customers
│   ├── 03_classification.ipynb         # Response Prediction (AUC=0.875)
│   ├── 04_clustering.ipynb             # Customer Segmentation (Sil=0.614)
│   └── 05_deep_learning.ipynb          # Neural Networks
│
├── 📦 src/                             # Reusable Python modules
│   ├── config.py                       # Hyperparameters & settings
│   ├── preprocessing.py                # Custom sklearn transformers
│   ├── models.py                       # Model factory functions
│   ├── evaluation.py                   # Metrics & threshold optimization
│   ├── visualization.py                # Plotting utilities
│   └── data_loader.py                  # Data loading
│
├── 📊 Data/
│   └── marketing_campaign.csv          # Customer dataset (2,240 records)
│
├── 📈 reports/                         # Analysis reports (Markdown)
│   ├── 00_executive_summary.md         # Cross-model comparison
│   ├── 01_eda_report.md
│   ├── 02_regression_analysis.md
│   ├── 03_classification_analysis.md
│   ├── 04_clustering_analysis.md
│   └── 05_deep_learning_analysis.md
│
├── 🤖 models/                          # Saved models (joblib/keras)
│   ├── best_regressor.joblib
│   ├── best_classifier.joblib
│   ├── kmeans_model.joblib
│   └── mlp_*.keras
│
├── requirements.txt
└── README.md
```

## 📊 Dataset

**Source**: [Kaggle - Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

| Property | Value |
|----------|-------|
| **Samples** | 2,240 customers |
| **Features** | 29 original → 20+ engineered |
| **Regression Target** | `TotalSpend_log` (log-transformed spending) |
| **Classification Target** | `Response` (15% positive - imbalanced) |

### Feature Categories

| Category | Features | Count |
|----------|----------|-------|
| 💰 **Value** | Income, TotalSpend | 2 |
| 🛒 **Products** | MntWines, MntMeatProducts, MntFruits, etc. | 6 |
| 📱 **Channels** | NumWebPurchases, NumCatalogPurchases, NumStorePurchases | 4 |
| 👤 **Demographics** | Age, Education, Marital_Status, Kidhome, Teenhome | 5 |
| ⏰ **Engagement** | Recency, NumWebVisitsMonth, CustomerTenure | 3 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/negin-kafee/Customer-Analytics-ML.git
cd Customer-Analytics-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. 01_eda.ipynb          → Understand the data
# 2. 02_regression.ipynb   → Predict spending
# 3. 03_classification.ipynb → Predict response
# 4. 04_clustering.ipynb   → Segment customers
# 5. 05_deep_learning.ipynb → Neural networks
```

---

## 📓 Notebook Details

### 1️⃣ Exploratory Data Analysis (`01_eda.ipynb`)
- Missing value analysis & imputation strategies
- Distribution analysis with statistical tests
- Correlation heatmaps and feature relationships
- **Anomaly Detection**: IQR, Isolation Forest, LOF

### 2️⃣ Regression Modeling (`02_regression.ipynb`)
**Goal**: Predict customer spending

| Model Category | Algorithms |
|----------------|------------|
| Linear | OLS, Ridge, Lasso, ElasticNet |
| Tree-based | Decision Tree, Random Forest, Extra Trees |
| Boosting | Gradient Boosting, **XGBoost** |
| SVM | Linear, RBF, Polynomial kernels |
| Neural | MLP Regressor |

**Best Result**: Random Forest with **R² = 0.970**

### 3️⃣ Classification Modeling (`03_classification.ipynb`)
**Goal**: Predict campaign response (imbalanced: 15% positive)

**Techniques**:
- Class weight balancing
- SMOTE oversampling
- Threshold optimization (cost-sensitive)
- Precision-Recall trade-off analysis

**Best Result**: Random Forest with **ROC-AUC = 0.875**, Recall = 78%

### 4️⃣ Clustering Analysis (`04_clustering.ipynb`)
**Goal**: Segment customers into actionable groups

**Optimization Journey**:
| Method | Silhouette | Notes |
|--------|------------|-------|
| Original (17 features) | 0.165 | Baseline |
| Log + StandardScaler | 0.466 | +182% |
| **MinMax + 2 Features** | **0.614** | **+272%** ✅ |

**Final Segments**:
| Segment | Size | Avg Spend | Response Rate |
|---------|------|-----------|---------------|
| 💎 VIP Champions | 13% | $1,753 | 40% |
| 🌟 High Value | 20% | $1,111 | 20% |
| 📊 Mid-Tier | 19% | $570 | 10% |
| 📉 Budget | 48% | $101 | 10% |

### 5️⃣ Deep Learning (`05_deep_learning.ipynb`)
**Goal**: Compare neural networks vs traditional ML

**Architecture**: 3-layer MLP with BatchNorm, Dropout (0.3), Early Stopping

| Task | MLP Performance | Best Traditional | Winner |
|------|-----------------|------------------|--------|
| Regression | R² = 0.953 | RF: R² = 0.970 | RF |
| Classification | AUC = 0.871 | RF: AUC = 0.875 | RF |

**Insight**: Tree-based models outperform deep learning on small tabular datasets (~2K samples)

---

## 🔧 Technical Highlights

### Custom sklearn Transformers

```python
# Located in src/preprocessing.py
from src.preprocessing import IQRCapper, LogTransformer, FeatureEngineer

# Example pipeline
pipeline = Pipeline([
    ('imputer', MedianImputer()),
    ('outlier_cap', IQRCapper(k=1.5)),
    ('log_transform', LogTransformer()),
    ('scaler', StandardScaler())
])
```

### Threshold Optimization

```python
# Cost-sensitive classification (src/evaluation.py)
from src.evaluation import find_cost_optimal_threshold

optimal_threshold, expected_cost, sweep_df = find_cost_optimal_threshold(
    y_true, y_proba, 
    c_fp=1.0,    # Cost of false positive
    c_fn=3.0     # Cost of false negative (3x more expensive)
)
```

### Model Comparison Framework

```python
# Centralized model training (src/models.py)
from src.models import get_regression_models, get_classification_models

models = get_regression_models()  # Returns dict of configured models
for name, model in models.items():
    model.fit(X_train, y_train)
    logger.log_regression(name, model, X_test, y_test)
```

---

## 📈 Key Findings

### 1. Feature Importance (Top 5)
| Rank | Feature | Impact |
|------|---------|--------|
| 1 | **Income** | Primary driver of spending |
| 2 | **Recency** | Key for campaign response |
| 3 | **MntWines** | Highest revenue category |
| 4 | **NumCatalogPurchases** | Signals engagement |
| 5 | **Age** | Correlates with spending |

### 2. Business Recommendations
- **Target VIP segment** (13% of customers → 38% of revenue)
- **Campaign focus**: Recency < 30 days = 3x response rate
- **Channel strategy**: Catalog for high-value, digital for budget

### 3. Model Selection Guide
| Scenario | Recommended | Why |
|----------|-------------|-----|
| Production (speed) | Random Forest | Fast inference, no GPU |
| Interpretability | Random Forest | Feature importances |
| Large data (>100K) | XGBoost or MLP | Better scaling |
| Maximum accuracy | Ensemble (RF + XGB) | Diversity helps |

---

## 📄 Reports

Detailed analysis reports are available in the `reports/` folder:

| Report | Description |
|--------|-------------|
| [Executive Summary](reports/00_executive_summary.md) | Cross-model comparison & recommendations |
| [EDA Report](reports/01_eda_report.md) | Data exploration findings |
| [Regression Analysis](reports/02_regression_analysis.md) | Spending prediction deep-dive |
| [Classification Analysis](reports/03_classification_analysis.md) | Response prediction analysis |
| [Clustering Analysis](reports/04_clustering_analysis.md) | Segmentation methodology & results |
| [Deep Learning Analysis](reports/05_deep_learning_analysis.md) | Neural network experiments |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+ |
| **ML Frameworks** | scikit-learn, XGBoost, TensorFlow/Keras |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn |
| **Notebooks** | Jupyter |

---

## 👤 Author

**Negin Kafee**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/negin-kafee)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) (Kaggle, CC0)
- **Inspiration**: Real-world marketing analytics challenges
- **Tools**: scikit-learn, XGBoost, TensorFlow communities

---

## ⭐ Star This Repo

If you found this project helpful, please consider giving it a star! It helps others discover it.

```
⭐ → Click the star button at the top right!
```
