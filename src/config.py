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
DATA_PATH = "data/raw/marketing_campaign.csv"
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

# Categorical features for modeling (KEEP ALL ORIGINAL CATEGORIES)
# Statistical analysis showed: Basic ≠ 2n Cycle (p<0.0001), Widow ≠ Single (p=0.02)
CAT_FEATURES = [
    "Education",            # All 5 levels: Basic, 2n Cycle, Graduation, Master, PhD
    "Marital_Status",       # 6 categories after cleaning: Married, Together, Single, Divorced, Widow, Other
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
# NOTE: Statistical analysis confirmed all 5 education levels have significantly
# different spending patterns. Basic ($82) vs 2n Cycle ($497) p<0.0001.
# DO NOT merge categories - it loses predictive information.
EDUCATION_MAPPING = {
    "Basic": 1,
    "2n Cycle": 2,
    "Graduation": 3,
    "Master": 4,
    "PhD": 5,
}

# All 5 education categories for one-hot encoding
EDUCATION_CATEGORIES = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]

# =============================================================================
# MARITAL STATUS CONSOLIDATION
# =============================================================================
# NOTE: Statistical analysis showed Widow ($739) is significantly different from
# Single ($606) - p=0.02. We only merge truly rare categories (n<10 total).
# Keep: Married, Together, Single, Divorced, Widow (all have 77+ samples)
MARITAL_STATUS_CONSOLIDATION = {
    "Alone": "Other",   # n=3
    "Absurd": "Other",  # n=2  
    "YOLO": "Other",    # n=2
}

# All 6 marital categories for one-hot encoding (after consolidation)
MARITAL_STATUS_CATEGORIES = ["Married", "Together", "Single", "Divorced", "Widow", "Other"]

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
]

# =============================================================================
# FEATURES TO EXCLUDE FROM CLASSIFICATION
# =============================================================================
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


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config() -> None:
    """
    Validate configuration settings at runtime.
    
    Raises
    ------
    ValueError
        If any configuration value is invalid.
    
    Example
    -------
    >>> from src.config import validate_config
    >>> validate_config()  # Raises if config is invalid
    """
    errors = []
    
    # Validate numeric ranges
    if not (0 < TEST_SIZE < 1):
        errors.append(f"TEST_SIZE must be between 0 and 1, got {TEST_SIZE}")
    
    if CV_FOLDS < 2:
        errors.append(f"CV_FOLDS must be at least 2, got {CV_FOLDS}")
    
    if RANDOM_STATE < 0:
        errors.append(f"RANDOM_STATE must be non-negative, got {RANDOM_STATE}")
    
    if ANOMALY_CONTAMINATION <= 0 or ANOMALY_CONTAMINATION >= 1:
        errors.append(f"ANOMALY_CONTAMINATION must be in (0, 1), got {ANOMALY_CONTAMINATION}")
    
    if IQR_MULTIPLIER <= 0:
        errors.append(f"IQR_MULTIPLIER must be positive, got {IQR_MULTIPLIER}")
    
    # Validate feature lists are not empty
    if not RAW_NUMERIC_FEATURES:
        errors.append("RAW_NUMERIC_FEATURES cannot be empty")
    
    if not SPENDING_COLUMNS:
        errors.append("SPENDING_COLUMNS cannot be empty")
    
    # Validate color codes (basic hex validation)
    import re
    hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
    for color_name, color_value in [
        ("MAIN_COLOR", MAIN_COLOR),
        ("SECONDARY_COLOR", SECONDARY_COLOR),
        ("ACCENT_COLOR", ACCENT_COLOR),
    ]:
        if not hex_pattern.match(color_value):
            errors.append(f"{color_name} must be a valid hex color, got {color_value}")
    
    # Raise all errors at once
    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


# Run validation on import (can be disabled for testing)
if __name__ != "__test__":
    try:
        validate_config()
    except ValueError as e:
        import warnings
        warnings.warn(f"Configuration validation warning: {e}", UserWarning)
