"""
================================================================================
Preprocessing Module — Custom Sklearn Transformers (Leak-Free)
================================================================================

This module contains custom sklearn-compatible transformers that ensure NO DATA
LEAKAGE by computing statistics (median, IQR bounds) only during fit() on
training data, then applying them during transform().

Key Transformers:
    1. MedianImputer: Imputes missing values using median (computed on train)
    2. IQRCapper: Caps outliers using IQR bounds (computed on train)
    3. LogTransformer: Applies log1p transformation
    4. FeatureEngineer: Creates derived features (Age, Tenure, TotalSpend)
    5. MaritalStatusMapper: Consolidates rare marital status categories
    6. EducationEncoder: Ordinal encoding for education levels

Why Custom Transformers?
    - sklearn's SimpleImputer computes median on training data automatically,
      but we need custom transformers for IQR capping and feature engineering
    - All transformers follow sklearn's fit/transform API for pipeline compatibility
    - This prevents the #1 data leakage mistake: computing statistics on full data

Usage:
    from src.preprocessing import MedianImputer, IQRCapper, get_preprocessor
    
    preprocessor = get_preprocessor(use_capped_features=False)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .config import (
    EDUCATION_MAPPING,
    MARITAL_STATUS_CONSOLIDATION,
    SPENDING_COLUMNS,
    IQR_MULTIPLIER,
    IQR_MULTIPLIER_CONSERVATIVE,
    NUM_FEATURES,
    ALL_CAT_FEATURES,
)


# =============================================================================
# CUSTOM TRANSFORMER: Median Imputer
# =============================================================================
class MedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using median computed from training data only.
    
    This prevents data leakage by:
    1. Computing median during fit() on training data
    2. Storing median and applying it during transform()
    
    Parameters
    ----------
    columns : list
        Column names to impute. If None, imputes all columns.
    
    Attributes
    ----------
    medians_ : dict
        Dictionary mapping column names to their median values.
    
    Example
    -------
    >>> imputer = MedianImputer(columns=['Income'])
    >>> imputer.fit(X_train)
    >>> X_train_imputed = imputer.transform(X_train)
    >>> X_test_imputed = imputer.transform(X_test)  # Uses training medians!
    """
    
    def __init__(self, columns=None):
        self.columns = columns
        self.medians_ = {}
    
    def fit(self, X, y=None):
        """Compute medians from training data."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        cols_to_impute = self.columns if self.columns else X.columns.tolist()
        
        for col in cols_to_impute:
            if col in X.columns:
                self.medians_[col] = X[col].median()
        
        return self
    
    def transform(self, X):
        """Apply median imputation using training medians."""
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, median_val in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for sklearn pipeline compatibility."""
        return input_features if input_features is not None else list(self.medians_.keys())


# =============================================================================
# CUSTOM TRANSFORMER: IQR Capper
# =============================================================================
class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers using IQR bounds computed from training data only.
    
    Outliers are defined as values below Q1 - k*IQR or above Q3 + k*IQR.
    Values outside these bounds are clipped (winsorized) to the bounds.
    
    This prevents data leakage by:
    1. Computing Q1, Q3, IQR during fit() on training data
    2. Storing bounds and applying them during transform()
    
    Parameters
    ----------
    columns : list
        Column names to cap.
    k : float, default=1.5
        IQR multiplier. Use 1.5 for standard, 3.0 for conservative.
    
    Attributes
    ----------
    bounds_ : dict
        Dictionary mapping column names to (lower_bound, upper_bound) tuples.
    
    Example
    -------
    >>> capper = IQRCapper(columns=['Income', 'Age'], k=1.5)
    >>> capper.fit(X_train)
    >>> X_capped = capper.transform(X_test)
    """
    
    def __init__(self, columns=None, k=1.5):
        self.columns = columns
        self.k = k
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        """Compute IQR bounds from training data."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        cols_to_cap = self.columns if self.columns else X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in cols_to_cap:
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - self.k * iqr
                upper_bound = q3 + self.k * iqr
                
                self.bounds_[col] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        """Apply IQR capping using training bounds."""
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for sklearn pipeline compatibility."""
        return input_features if input_features is not None else list(self.bounds_.keys())


# =============================================================================
# CUSTOM TRANSFORMER: Log Transformer
# =============================================================================
class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation to specified columns.
    
    log1p(x) = log(1 + x) handles zero values gracefully.
    This is useful for right-skewed distributions like Income and Spending.
    
    Parameters
    ----------
    columns : list
        Column names to transform. Creates new columns with '_log' suffix.
    
    Example
    -------
    >>> log_transformer = LogTransformer(columns=['Income', 'TotalSpend'])
    >>> X_transformed = log_transformer.fit_transform(X)
    >>> # Creates 'Income_log' and 'TotalSpend_log' columns
    """
    
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        """No fitting required - log1p is a stateless transformation."""
        return self
    
    def transform(self, X):
        """Apply log1p transformation, creating new columns."""
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        
        if self.columns:
            for col in self.columns:
                if col in X.columns:
                    X[f"{col}_log"] = np.log1p(X[col])
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names including new log columns."""
        if input_features is None:
            return None
        output_features = list(input_features)
        if self.columns:
            for col in self.columns:
                output_features.append(f"{col}_log")
        return output_features


# =============================================================================
# CUSTOM TRANSFORMER: Feature Engineer
# =============================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived features from raw data.
    
    Features Created:
    - Age: Computed from Year_Birth and reference year (from Dt_Customer)
    - Tenure_Days: Days since customer enrollment
    - TotalSpend: Sum of all spending columns
    - Education_Level: Ordinal encoding of Education
    - Cleaned Marital_Status: Rare categories consolidated to "Other"
    
    This transformer is STATEFUL:
    - Reference date/year computed from training data during fit()
    - Applied consistently to test data during transform()
    
    Parameters
    ----------
    create_log_features : bool, default=True
        Whether to create log-transformed versions of skewed features.
    
    Attributes
    ----------
    reference_date_ : pd.Timestamp
        Maximum date in training data (for Tenure calculation)
    reference_year_ : int
        Year from reference_date_ (for Age calculation)
    """
    
    def __init__(self, create_log_features=True):
        self.create_log_features = create_log_features
        self.reference_date_ = None
        self.reference_year_ = None
    
    def fit(self, X, y=None):
        """Compute reference date/year from training data."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        if "Dt_Customer" in X.columns:
            dates = pd.to_datetime(X["Dt_Customer"], dayfirst=True)
            self.reference_date_ = dates.max()
            self.reference_year_ = dates.dt.year.max()
        else:
            # Default fallback
            self.reference_date_ = pd.Timestamp("2014-12-31")
            self.reference_year_ = 2014
        
        return self
    
    def transform(self, X):
        """Create engineered features using training reference values."""
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        
        # --- Age from Year_Birth ---
        if "Year_Birth" in X.columns:
            X["Age"] = self.reference_year_ - X["Year_Birth"]
        
        # --- Tenure_Days from Dt_Customer ---
        if "Dt_Customer" in X.columns:
            X["Dt_Customer"] = pd.to_datetime(X["Dt_Customer"], dayfirst=True)
            X["Tenure_Days"] = (self.reference_date_ - X["Dt_Customer"]).dt.days
        
        # --- TotalSpend from spending columns ---
        spending_cols_present = [c for c in SPENDING_COLUMNS if c in X.columns]
        if spending_cols_present:
            X["TotalSpend"] = X[spending_cols_present].sum(axis=1)
        
        # --- Education_Level (ordinal encoding) ---
        if "Education" in X.columns:
            X["Education_Level"] = X["Education"].map(EDUCATION_MAPPING)
        
        # --- Marital_Status consolidation ---
        if "Marital_Status" in X.columns:
            X["Marital_Status"] = X["Marital_Status"].replace(MARITAL_STATUS_CONSOLIDATION)
        
        # --- Log transformations ---
        if self.create_log_features:
            if "Income" in X.columns:
                X["Income_log"] = np.log1p(X["Income"])
            if "TotalSpend" in X.columns:
                X["TotalSpend_log"] = np.log1p(X["TotalSpend"])
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names including engineered features."""
        new_features = ["Age", "Tenure_Days", "TotalSpend", "Education_Level"]
        if self.create_log_features:
            new_features.extend(["Income_log", "TotalSpend_log"])
        
        if input_features is None:
            return new_features
        
        return list(input_features) + new_features


# =============================================================================
# CUSTOM TRANSFORMER: Capped Feature Creator
# =============================================================================
class CappedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates IQR-capped versions of specified features.
    
    This is useful for comparing model performance with and without
    outlier treatment. Creates new columns with '_capped' suffix.
    
    Parameters
    ----------
    columns_k : dict
        Dictionary mapping column names to IQR multiplier k.
        Example: {'Income': 3.0, 'Age': 1.5}
    
    Attributes
    ----------
    bounds_ : dict
        Dictionary mapping column names to (lower_bound, upper_bound) tuples.
    """
    
    def __init__(self, columns_k=None):
        self.columns_k = columns_k or {"Income": IQR_MULTIPLIER_CONSERVATIVE, "Age": IQR_MULTIPLIER}
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        """Compute IQR bounds from training data."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col, k in self.columns_k.items():
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                
                self.bounds_[col] = (q1 - k * iqr, q3 + k * iqr)
        
        return self
    
    def transform(self, X):
        """Create capped features using training bounds."""
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[f"{col}_capped"] = X[col].clip(lower=lower, upper=upper)
                
                # Also create log version for income
                if col == "Income":
                    X["Income_capped_log"] = np.log1p(X[f"{col}_capped"])
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names including capped columns."""
        capped_features = [f"{col}_capped" for col in self.bounds_.keys()]
        if "Income" in self.bounds_:
            capped_features.append("Income_capped_log")
        
        if input_features is None:
            return capped_features
        
        return list(input_features) + capped_features


# =============================================================================
# PREPROCESSING PIPELINE FACTORY
# =============================================================================
def get_full_preprocessor():
    """
    Returns a complete preprocessing pipeline for initial data preparation.
    
    This pipeline handles:
    1. Missing value imputation (Income median)
    2. Feature engineering (Age, Tenure, TotalSpend, encodings)
    3. Capped feature creation (for outlier-sensitive models)
    
    Returns
    -------
    Pipeline
        Sklearn pipeline for full preprocessing.
    
    Example
    -------
    >>> preprocessor = get_full_preprocessor()
    >>> df_processed = preprocessor.fit_transform(df_train)
    """
    return Pipeline([
        ("imputer", MedianImputer(columns=["Income"])),
        ("engineer", FeatureEngineer(create_log_features=True)),
        ("capper", CappedFeatureCreator(columns_k={"Income": 3.0, "Age": 1.5})),
    ])


def get_model_preprocessor(num_features, cat_features, scale_numeric=True):
    """
    Returns a ColumnTransformer for model-ready preprocessing.
    
    This preprocessor handles:
    1. Numeric features: StandardScaler (optional)
    2. Categorical features: OneHotEncoder with rare category handling
    
    Parameters
    ----------
    num_features : list
        Numeric feature names.
    cat_features : list
        Categorical feature names.
    scale_numeric : bool, default=True
        Whether to apply StandardScaler to numeric features.
    
    Returns
    -------
    ColumnTransformer
        Preprocessor for model training.
    
    Example
    -------
    >>> preprocessor = get_model_preprocessor(NUM_FEATURES, ALL_CAT_FEATURES)
    >>> pipe = Pipeline([('pre', preprocessor), ('model', LinearRegression())])
    """
    
    # Numeric pipeline
    if scale_numeric:
        num_pipeline = StandardScaler()
    else:
        num_pipeline = "passthrough"
    
    # Categorical pipeline with rare category handling
    cat_pipeline = OneHotEncoder(
        drop="first",
        handle_unknown="infrequent_if_exist",
        min_frequency=50,  # Rare categories with <50 samples grouped
        sparse_output=False,
    )
    
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",  # Drop any columns not specified
        verbose_feature_names_out=False,
    )


def get_tree_preprocessor(cat_features):
    """
    Returns a ColumnTransformer for tree-based models (no scaling needed).
    
    Tree-based models (Decision Tree, Random Forest, XGBoost) are scale-invariant,
    so we only need to encode categorical features.
    
    Parameters
    ----------
    cat_features : list
        Categorical feature names.
    
    Returns
    -------
    ColumnTransformer
        Preprocessor for tree-based models.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(
                drop="first",
                handle_unknown="infrequent_if_exist",
                min_frequency=50,
                sparse_output=False,
            ), cat_features),
        ],
        remainder="passthrough",  # Keep numeric columns as-is
        verbose_feature_names_out=False,
    )


def get_clustering_preprocessor(features, use_log=True):
    """
    Returns a preprocessing pipeline for clustering.
    
    Clustering with K-Means uses Euclidean distance, so:
    1. Log transformation reduces skewness (optional)
    2. StandardScaler ensures equal feature weights
    
    Parameters
    ----------
    features : list
        Feature names to use for clustering.
    use_log : bool, default=True
        Whether to apply log1p transformation before scaling.
    
    Returns
    -------
    Pipeline
        Preprocessing pipeline for clustering.
    """
    steps = []
    
    if use_log:
        steps.append(("log", LogTransformer(columns=features)))
    
    steps.append(("scaler", StandardScaler()))
    
    return Pipeline(steps)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def detect_outliers_iqr(df, columns, k=1.5):
    """
    Detect outliers using IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list
        Columns to check for outliers.
    k : float, default=1.5
        IQR multiplier.
    
    Returns
    -------
    pd.DataFrame
        Summary of outlier detection results.
    """
    results = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        
        outliers = series[(series < lower) | (series > upper)]
        
        results.append({
            "feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": len(outliers),
            "pct_outliers": len(outliers) / len(series) * 100,
            "min_outlier": outliers.min() if len(outliers) > 0 else np.nan,
            "max_outlier": outliers.max() if len(outliers) > 0 else np.nan,
        })
    
    return pd.DataFrame(results).sort_values("pct_outliers", ascending=False)


def get_anomaly_scores(X, methods=["zscore", "isolation_forest", "lof"], contamination=0.05):
    """
    Compute anomaly scores using multiple methods.
    
    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    methods : list
        Methods to use: "zscore", "isolation_forest", "lof"
    contamination : float
        Expected proportion of outliers.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with anomaly scores from each method.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    
    results = {}
    
    if "zscore" in methods:
        results["zscore"] = np.abs(X).mean(axis=1)
    
    if "isolation_forest" in methods:
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X)
        results["isolation_forest"] = -iso.score_samples(X)
    
    if "lof" in methods:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        lof.fit_predict(X)
        results["lof"] = -lof.negative_outlier_factor_
    
    return pd.DataFrame(results)
