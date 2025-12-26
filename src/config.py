"""
================================================================================
Configuration Module — Global Constants & Feature Definitions
================================================================================

This module centralizes all configuration settings for the Customer Personality
Analysis project. By defining constants once here, we ensure consistency across
all notebooks and prevent errors from mismatched feature lists.

Key Design Decisions:
    1. RANDOM_STATE = 42 ensures reproducibility across all experiments
    2. Feature lists are defined once to prevent data leakage from inconsistency
    3. Color scheme provides consistent visualization aesthetics
    4. Target variables are explicitly defined for each task type

Usage:
    from src.config import RANDOM_STATE, NUM_FEATURES, CAT_FEATURES, TARGET_REGRESSION
"""

import numpy as np

# =============================================================================
# REPRODUCIBILITY SETTINGS
# =============================================================================
RANDOM_STATE = 42  # Ensures reproducible results across all experiments
TEST_SIZE = 0.20   # 80/20 train/test split - industry standard
CV_FOLDS = 5       # 5-fold cross-validation balances bias/variance tradeoff

# =============================================================================
# DATA PATHS
# =============================================================================
DATA_PATH = "Data/marketing_campaign.csv"
DATA_SEPARATOR = "\t"  # Tab-separated values
MODELS_DIR = "models/"
PROCESSED_DATA_DIR = "data/processed/"

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
# These are the RAW features from the dataset before any engineering

# Original numeric features (before transformation)
RAW_NUMERIC_FEATURES = [
    "Income",
    "Year_Birth",
    "Kidhome",
    "Teenhome",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

# Original categorical features
RAW_CATEGORICAL_FEATURES = [
    "Education",
    "Marital_Status",
]

# Binary features (campaign responses and complaints)
BINARY_FEATURES = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
    "Complain",
]

# Date feature
DATE_FEATURE = "Dt_Customer"

# ID column (to be dropped for modeling)
ID_COLUMN = "ID"

# Constant columns (to be dropped - no variance)
CONSTANT_COLUMNS = ["Z_CostContact", "Z_Revenue"]

# =============================================================================
# ENGINEERED FEATURE DEFINITIONS
# =============================================================================
# Features created during preprocessing

# Spending columns (to create TotalSpend)
SPENDING_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

# Features to apply log transformation (right-skewed distributions)
LOG_TRANSFORM_FEATURES = ["Income", "TotalSpend"]

# Features to apply IQR capping (outlier treatment)
IQR_CAP_FEATURES = ["Income", "Age"]

# =============================================================================
# MODEL-READY FEATURE SETS (after preprocessing)
# =============================================================================

# Numeric features for regression/classification (after engineering)
NUM_FEATURES = [
    "Income_log",           # Log-transformed income
    "Age",                  # Calculated from Year_Birth
    "Tenure_Days",          # Days since customer enrollment
    "Kidhome",
    "Teenhome",
    "Recency",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

# Alternative: Capped features for outlier-sensitive models
NUM_FEATURES_CAPPED = [
    "Income_capped_log",    # Log of IQR-capped income
    "Age_capped",           # IQR-capped age
    "Tenure_Days",
    "Kidhome",
    "Teenhome",
    "Recency",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

# Categorical features for modeling
CAT_FEATURES = [
    "Education_Level",      # Ordinal encoded: Basic=1 to PhD=5
    "Marital_Status",       # Consolidated: rare categories → "Other"
]

# Binary features to include as categorical
CAT_BINARY_FEATURES = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
    "Complain",
]

# All categorical features for preprocessing
ALL_CAT_FEATURES = CAT_FEATURES + CAT_BINARY_FEATURES

# Lasso-selected features (for reduced models)
LASSO_SELECTED_FEATURES = [
    "NumCatalogPurchases",
    "Income_log",
    "NumWebPurchases",
    "NumStorePurchases",
    "Kidhome",
    "NumDealsPurchases",
    "Teenhome",
    "Tenure_Days",
]

# =============================================================================
# TARGET VARIABLES
# =============================================================================

# Regression target
TARGET_REGRESSION = "TotalSpend_log"  # Log-transformed total spending
TARGET_REGRESSION_RAW = "TotalSpend"  # Raw total spending

# Classification target
TARGET_CLASSIFICATION = "Response"    # Campaign response (binary: 0/1)

# =============================================================================
# CLUSTERING FEATURES (RFM-based)
# =============================================================================
CLUSTERING_FEATURES = [
    "Recency",              # R: Days since last purchase
    "NumWebPurchases",      # F: Frequency (web)
    "NumCatalogPurchases",  # F: Frequency (catalog)
    "NumStorePurchases",    # F: Frequency (store)
    "TotalSpend",           # M: Monetary value
    "Income",               # Economic capacity
]

# =============================================================================
# EDUCATION MAPPING (Ordinal Encoding)
# =============================================================================
EDUCATION_MAPPING = {
    "Basic": 1,
    "2n Cycle": 2,
    "Graduation": 3,
    "Master": 4,
    "PhD": 5,
}

# Marital status consolidation (rare categories → Other)
MARITAL_STATUS_CONSOLIDATION = {
    "Alone": "Other",
    "Absurd": "Other",
    "YOLO": "Other",
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
# Color scheme for consistent plots
MAIN_COLOR = "#B388EB"          # Primary purple
SECONDARY_COLOR = "#63AFA8"     # Teal for comparison
ACCENT_COLOR = "#F4A261"        # Orange accent
POSITIVE_COLOR = "#2A9D8F"      # Green for positive values
NEGATIVE_COLOR = "#E76F51"      # Red for negative values

# Plot settings
FIGURE_DPI = 100
FIGURE_SIZE_SMALL = (6, 4)
FIGURE_SIZE_MEDIUM = (10, 6)
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_SIZE_WIDE = (16, 6)

# =============================================================================
# MODEL HYPERPARAMETER GRIDS
# =============================================================================

# Ridge/Lasso alpha search space
REGULARIZATION_ALPHAS = np.logspace(-4, 2, 50)

# Decision Tree hyperparameters
TREE_PARAM_GRID = {
    "max_depth": [4, 6, 8, 10, 12, None],
    "min_samples_leaf": [1, 2, 4, 6, 8],
    "min_samples_split": [2, 5, 10],
}

# Random Forest hyperparameters
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# XGBoost hyperparameters
XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# SVR hyperparameters
SVR_PARAM_GRID = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1, 0.01],
}

# =============================================================================
# ANOMALY DETECTION SETTINGS
# =============================================================================
ANOMALY_CONTAMINATION = 0.05  # Expect ~5% outliers
IQR_MULTIPLIER = 1.5          # Standard IQR multiplier for outlier detection
IQR_MULTIPLIER_CONSERVATIVE = 3.0  # Conservative for income (highly skewed)

# =============================================================================
# CLUSTERING SETTINGS
# =============================================================================
CLUSTER_K_RANGE = range(2, 11)  # Test k from 2 to 10
GMM_COV_TYPES = ["full", "tied", "diag", "spherical"]

# =============================================================================
# CLASSIFICATION SETTINGS
# =============================================================================
# Cost matrix for asymmetric misclassification costs
# In marketing: missing a potential customer (FN) is often more costly than
# contacting someone who won't respond (FP)
COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 3.0  # 3x more costly to miss a potential subscriber

# Threshold search space
THRESHOLD_RANGE = np.linspace(0.01, 0.99, 99)

# =============================================================================
# DEEP LEARNING SETTINGS
# =============================================================================
DL_HIDDEN_LAYERS = [256, 128]
DL_DROPOUT_RATE = 0.3
DL_LEARNING_RATE = 1e-3
DL_BATCH_SIZE = 128
DL_MAX_EPOCHS = 50
DL_EARLY_STOPPING_PATIENCE = 5

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# Shorter names used in notebooks
RAW_NUM_COLS = RAW_NUMERIC_FEATURES
RAW_CAT_COLS = RAW_CATEGORICAL_FEATURES
SPENDING_COLS = SPENDING_COLUMNS
PURCHASE_COLS = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]
CAMPAIGN_COLS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
]

# =============================================================================
# DATA LEAKAGE WARNING - FEATURES TO EXCLUDE
# =============================================================================
# These features have data leakage for classification (Response prediction)
# They are derived from or perfectly correlated with campaign responses
LEAKY_FEATURES = [
    "TotalAccepted",
    "IsPreviousResponder",
    "AcceptedCmp1",
    "AcceptedCmp2", 
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]

# =============================================================================
# ENGINEERED FEATURES
# =============================================================================
ENGINEERED_NUMERIC = [
    "Age",
    "Tenure_Days",
    "Tenure_Months",
    "TotalSpend",
    "TotalSpend_log",
    "SpendingRatio",
    "TotalPurchases",
    "AvgSpendPerPurchase",
    "WebPurchaseRatio",
    "FamilySize",
    "TotalChildren",
    "IncomePerCapita",
]

ENGINEERED_BINARY = [
    "HasChildren",
    "IsPartner",
    "IsHighSpender",
    "IsPreviousResponder",  # Note: Exclude for classification!
]

ENGINEERED_CATEGORICAL = [
    "Education_Level",
    "Education_Num",
    "RecencyCategory",
]
