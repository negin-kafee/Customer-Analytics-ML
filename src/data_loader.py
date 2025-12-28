"""
================================================================================
Data Loader Module — Dataset Loading and Splitting
================================================================================

This module provides:
- Data loading functions with validation
- Train/test splitting with proper stratification
- Data overview and summary statistics

Design Principles:
    1. Single source of truth for data loading
    2. Built-in validation and sanity checks
    3. Consistent splitting strategy
    4. Reproducibility through fixed random states

Usage:
    from src.data_loader import load_data, split_data
    
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union

from sklearn.model_selection import train_test_split

from .config import (
    DATA_PATH,
    DATA_SEPARATOR,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    RAW_NUM_COLS,
    RAW_CAT_COLS,
)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(filepath: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load the marketing campaign dataset.
    
    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. If None, uses DATA_PATH from config.
    verbose : bool
        Print summary info after loading.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    
    Raises
    ------
    FileNotFoundError
        If data file does not exist.
    ValueError
        If data has unexpected format.
    """
    filepath = filepath or DATA_PATH
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Expected path: {DATA_PATH}"
        )
    
    # Load with configured separator
    df = pd.read_csv(filepath, sep=DATA_SEPARATOR)
    
    if verbose:
        print(f"✓ Loaded data from: {filepath}")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate dataset and check for common issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    dict
        Validation results.
    """
    results = {
        "shape": df.shape,
        "n_duplicates": df.duplicated().sum(),
        "missing_values": df.isnull().sum().sum(),
        "missing_by_column": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime64"]).columns.tolist(),
    }
    
    return results


def data_overview(df: pd.DataFrame) -> None:
    """
    Print comprehensive data overview.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    """
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    
    print(f"\n📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Memory usage
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"💾 Memory: {mem_mb:.2f} MB")
    
    # Data types
    print(f"\n📋 Column Types:")
    for dtype in df.dtypes.value_counts().index:
        count = (df.dtypes == dtype).sum()
        print(f"   {dtype}: {count}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        print(f"\n⚠️  Missing Values ({len(missing_cols)} columns):")
        for col, count in missing_cols.items():
            pct = 100 * count / len(df)
            print(f"   {col}: {count:,} ({pct:.1f}%)")
    else:
        print("\n✓ No missing values")
    
    # Duplicates
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        print(f"\n⚠️  Duplicates: {n_dups:,} rows")
    else:
        print("\n✓ No duplicate rows")
    
    # Numeric summary
    print("\n📈 Numeric Columns Summary:")
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.describe().T[["count", "mean", "std", "min", "max"]].to_string())
    
    print("\n" + "=" * 60)


def get_feature_types(df: pd.DataFrame) -> dict:
    """
    Categorize features by type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    dict
        Feature lists by type.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Identify binary columns (0/1 or two unique values)
    binary_cols = [
        col for col in numeric_cols 
        if df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1})
    ]
    
    # Identify high cardinality categorical
    high_cardinality = [
        col for col in categorical_cols 
        if df[col].nunique() > 10
    ]
    
    return {
        "numeric": [c for c in numeric_cols if c not in binary_cols],
        "binary": binary_cols,
        "categorical": categorical_cols,
        "high_cardinality": high_cardinality
    }


# =============================================================================
# DATA SPLITTING
# =============================================================================
def split_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = False
) -> Tuple:
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    test_size : float
        Proportion of data for testing.
    random_state : int
        Random state for reproducibility.
    stratify : bool
        Whether to stratify split (for classification).
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    print(f"✓ Data split complete:")
    print(f"  Train: {len(X_train):,} samples ({100*(1-test_size):.0f}%)")
    print(f"  Test:  {len(X_test):,} samples ({100*test_size:.0f}%)")
    
    if stratify and hasattr(y, "value_counts"):
        print(f"  Stratified by target (class distribution preserved)")
    
    return X_train, X_test, y_train, y_test


def prepare_regression_data(df: pd.DataFrame, target: str = TARGET_REGRESSION) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for regression task.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (after feature engineering).
    target : str
        Target column name.
    
    Returns
    -------
    tuple
        (X, y)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame. "
                        f"Make sure feature engineering has been applied.")
    
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    
    # Drop any remaining target-related columns that could leak
    leak_cols = [c for c in X.columns if "TotalSpend" in c or "Spend" in c]
    if leak_cols:
        print(f"⚠️  Removing potential leakage columns: {leak_cols}")
        X = X.drop(columns=leak_cols)
    
    print(f"✓ Regression data prepared:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target: {target}")
    
    return X, y


def prepare_classification_data(df: pd.DataFrame, target: str = TARGET_CLASSIFICATION) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for classification task.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Target column name.
    
    Returns
    -------
    tuple
        (X, y)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")
    
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    
    # Check class balance
    class_counts = y.value_counts()
    print(f"✓ Classification data prepared:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target: {target}")
    print(f"  Class distribution:")
    for cls, count in class_counts.items():
        pct = 100 * count / len(y)
        print(f"    Class {cls}: {count:,} ({pct:.1f}%)")
    
    return X, y


def prepare_clustering_data(
    df: pd.DataFrame,
    features: List[str],
    scale: bool = True
) -> np.ndarray:
    """
    Prepare data for clustering task.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    features : list
        Feature columns to use.
    scale : bool
        Whether to standardize features.
    
    Returns
    -------
    np.ndarray
        Feature matrix ready for clustering.
    """
    from sklearn.preprocessing import StandardScaler
    
    X = df[features].copy()
    
    # Check for missing values
    if X.isnull().any().any():
        print("⚠️  Missing values detected, filling with median")
        X = X.fillna(X.median())
    
    X_array = X.values
    
    if scale:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)
        print("✓ Features standardized (mean=0, std=1)")
    
    print(f"✓ Clustering data prepared:")
    print(f"  Samples: {X_array.shape[0]:,}")
    print(f"  Features: {X_array.shape[1]}")
    
    return X_array


# =============================================================================
# DATA CLEANING
# =============================================================================
def clean_income(df: pd.DataFrame, income_col: str = "Income") -> pd.DataFrame:
    """
    Handle missing income values.
    
    Note: Does NOT impute here - imputation happens in preprocessing pipeline
    to avoid data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    income_col : str
        Income column name.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with income issues flagged.
    """
    df = df.copy()
    
    if income_col in df.columns:
        missing = df[income_col].isnull().sum()
        if missing > 0:
            print(f"ℹ️  {income_col}: {missing} missing values will be imputed during preprocessing")
    
    return df


def clean_dates(df: pd.DataFrame, date_col: str = "Dt_Customer") -> pd.DataFrame:
    """
    Parse and clean date columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Date column to parse.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed dates.
    """
    df = df.copy()
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format="%d-%m-%Y", errors="coerce")
        print(f"✓ Parsed {date_col} to datetime")
    
    return df


def remove_outlier_rows(df: pd.DataFrame, column: str, lower_pct: float = 0, upper_pct: float = 99) -> pd.DataFrame:
    """
    Remove rows with extreme outliers in specified column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to check for outliers.
    lower_pct : float
        Lower percentile cutoff.
    upper_pct : float
        Upper percentile cutoff.
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    lower = df[column].quantile(lower_pct / 100)
    upper = df[column].quantile(upper_pct / 100)
    
    mask = (df[column] >= lower) & (df[column] <= upper)
    n_removed = (~mask).sum()
    
    if n_removed > 0:
        print(f"ℹ️  Removed {n_removed} rows with {column} outside [{lower:.1f}, {upper:.1f}]")
    
    return df[mask].copy()
