"""
Pytest Configuration and Fixtures
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'Income': np.random.lognormal(10.5, 0.5, n_samples),
        'Year_Birth': np.random.randint(1940, 2000, n_samples),
        'Kidhome': np.random.randint(0, 3, n_samples),
        'Teenhome': np.random.randint(0, 3, n_samples),
        'Recency': np.random.randint(0, 100, n_samples),
        'MntWines': np.random.lognormal(5, 1, n_samples),
        'MntFruits': np.random.lognormal(3, 1, n_samples),
        'MntMeatProducts': np.random.lognormal(4, 1, n_samples),
        'MntFishProducts': np.random.lognormal(3, 1, n_samples),
        'MntSweetProducts': np.random.lognormal(3, 1, n_samples),
        'MntGoldProds': np.random.lognormal(3, 1, n_samples),
        'NumDealsPurchases': np.random.randint(0, 15, n_samples),
        'NumWebPurchases': np.random.randint(0, 20, n_samples),
        'NumCatalogPurchases': np.random.randint(0, 20, n_samples),
        'NumStorePurchases': np.random.randint(0, 15, n_samples),
        'NumWebVisitsMonth': np.random.randint(0, 20, n_samples),
        'Education': np.random.choice(['Basic', 'Graduation', 'PhD', 'Master'], n_samples),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Together', 'Alone', 'YOLO'], n_samples),
        'Dt_Customer': pd.date_range('2012-01-01', periods=n_samples, freq='10D').strftime('%d-%m-%Y'),
        'Response': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'AcceptedCmp1': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'AcceptedCmp2': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'AcceptedCmp3': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'AcceptedCmp4': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'AcceptedCmp5': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Complain': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
    })


@pytest.fixture
def sample_dataframe_with_missing(sample_dataframe):
    """Create a sample DataFrame with missing values."""
    df = sample_dataframe.copy()
    # Add some missing values
    df.loc[0:4, 'Income'] = np.nan
    df.loc[10:12, 'Education'] = np.nan
    return df


@pytest.fixture
def sample_features():
    """Create sample feature matrix."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100) * 10,
        'feature3': np.random.randn(100) + 5,
    })


@pytest.fixture
def sample_target_regression():
    """Create sample regression target."""
    np.random.seed(42)
    return pd.Series(np.random.lognormal(5, 1, 100))


@pytest.fixture
def sample_target_classification():
    """Create sample classification target."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100, p=[0.85, 0.15]))
