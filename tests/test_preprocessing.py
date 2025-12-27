"""
Unit Tests for Preprocessing Module
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    MedianImputer,
    IQRCapper,
    LogTransformer,
    FeatureEngineer,
    CappedFeatureCreator,
    detect_outliers_iqr,
)


class TestMedianImputer:
    """Tests for MedianImputer transformer."""
    
    def test_fit_stores_medians(self, sample_dataframe):
        """Test that fit() computes and stores median values."""
        imputer = MedianImputer(columns=['Income'])
        imputer.fit(sample_dataframe)
        
        assert 'Income' in imputer.medians_
        assert imputer.medians_['Income'] == sample_dataframe['Income'].median()
    
    def test_transform_fills_missing(self, sample_dataframe_with_missing):
        """Test that transform() fills missing values with stored medians."""
        df = sample_dataframe_with_missing.copy()
        original_missing = df['Income'].isna().sum()
        assert original_missing > 0, "Test data should have missing values"
        
        imputer = MedianImputer(columns=['Income'])
        imputer.fit(df)
        transformed = imputer.transform(df)
        
        assert transformed['Income'].isna().sum() == 0
    
    def test_no_data_leakage(self, sample_dataframe):
        """Test that transform uses training medians, not test data medians."""
        # Split data
        train_df = sample_dataframe.iloc[:50].copy()
        test_df = sample_dataframe.iloc[50:].copy()
        
        # Add missing values to test set
        test_df.loc[test_df.index[0], 'Income'] = np.nan
        
        # Fit on train, transform test
        imputer = MedianImputer(columns=['Income'])
        imputer.fit(train_df)
        transformed_test = imputer.transform(test_df)
        
        # The imputed value should be train median, not test median
        train_median = train_df['Income'].median()
        assert transformed_test.loc[test_df.index[0], 'Income'] == train_median


class TestIQRCapper:
    """Tests for IQRCapper transformer."""
    
    def test_fit_computes_bounds(self, sample_dataframe):
        """Test that fit() computes IQR bounds."""
        capper = IQRCapper(columns=['Income'], k=1.5)
        capper.fit(sample_dataframe)
        
        assert 'Income' in capper.bounds_
        lower, upper = capper.bounds_['Income']
        assert lower < upper
    
    def test_transform_clips_outliers(self):
        """Test that outliers are clipped to bounds."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        capper = IQRCapper(columns=['value'], k=1.5)
        capper.fit(df)
        transformed = capper.transform(df)
        
        # The outlier should be clipped
        assert transformed['value'].max() < 100
    
    def test_non_outliers_unchanged(self):
        """Test that non-outlier values are not changed."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]  # No outliers
        })
        
        capper = IQRCapper(columns=['value'], k=1.5)
        capper.fit(df)
        transformed = capper.transform(df)
        
        # Values should be unchanged
        pd.testing.assert_series_equal(df['value'], transformed['value'])


class TestLogTransformer:
    """Tests for LogTransformer."""
    
    def test_creates_log_columns(self, sample_dataframe):
        """Test that log columns are created."""
        transformer = LogTransformer(columns=['Income'])
        transformed = transformer.fit_transform(sample_dataframe)
        
        assert 'Income_log' in transformed.columns
    
    def test_log_values_correct(self):
        """Test that log1p transformation is applied correctly."""
        df = pd.DataFrame({'value': [0, 1, 10, 100]})
        
        transformer = LogTransformer(columns=['value'])
        transformed = transformer.fit_transform(df)
        
        expected = np.log1p(df['value'])
        pd.testing.assert_series_equal(
            transformed['value_log'], 
            expected, 
            check_names=False
        )
    
    def test_handles_zeros(self):
        """Test that log1p handles zeros correctly."""
        df = pd.DataFrame({'value': [0, 0, 1, 2]})
        
        transformer = LogTransformer(columns=['value'])
        transformed = transformer.fit_transform(df)
        
        # log1p(0) = 0
        assert transformed['value_log'].iloc[0] == 0


class TestFeatureEngineer:
    """Tests for FeatureEngineer transformer."""
    
    def test_creates_age_feature(self, sample_dataframe):
        """Test that Age is computed from Year_Birth."""
        engineer = FeatureEngineer()
        transformed = engineer.fit_transform(sample_dataframe)
        
        assert 'Age' in transformed.columns
        assert transformed['Age'].min() > 0
    
    def test_creates_tenure_feature(self, sample_dataframe):
        """Test that Tenure_Days is computed from Dt_Customer."""
        engineer = FeatureEngineer()
        transformed = engineer.fit_transform(sample_dataframe)
        
        assert 'Tenure_Days' in transformed.columns
        assert transformed['Tenure_Days'].min() >= 0
    
    def test_creates_total_spend(self, sample_dataframe):
        """Test that TotalSpend is computed."""
        engineer = FeatureEngineer()
        transformed = engineer.fit_transform(sample_dataframe)
        
        assert 'TotalSpend' in transformed.columns
        assert transformed['TotalSpend'].min() >= 0
    
    def test_creates_log_features(self, sample_dataframe):
        """Test that log features are created when enabled."""
        engineer = FeatureEngineer(create_log_features=True)
        transformed = engineer.fit_transform(sample_dataframe)
        
        assert 'Income_log' in transformed.columns
        assert 'TotalSpend_log' in transformed.columns
    
    def test_no_log_features_when_disabled(self, sample_dataframe):
        """Test that log features are not created when disabled."""
        engineer = FeatureEngineer(create_log_features=False)
        transformed = engineer.fit_transform(sample_dataframe)
        
        assert 'Income_log' not in transformed.columns
        assert 'TotalSpend_log' not in transformed.columns


class TestDetectOutliersIQR:
    """Tests for detect_outliers_iqr function."""
    
    def test_detects_outliers(self):
        """Test that outliers are detected."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        result = detect_outliers_iqr(df, columns=['value'], k=1.5)
        
        assert len(result) == 1
        assert result.iloc[0]['n_outliers'] > 0
    
    def test_no_false_positives(self):
        """Test that normal data has no outliers."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]  # No outliers
        })
        
        result = detect_outliers_iqr(df, columns=['value'], k=1.5)
        
        assert result.iloc[0]['n_outliers'] == 0


class TestPipelineIntegration:
    """Tests for sklearn pipeline integration."""
    
    def test_transformers_work_in_pipeline(self, sample_dataframe):
        """Test that transformers can be used in sklearn Pipeline."""
        pipeline = Pipeline([
            ('imputer', MedianImputer(columns=['Income'])),
            ('engineer', FeatureEngineer(create_log_features=True)),
        ])
        
        # Should not raise
        transformed = pipeline.fit_transform(sample_dataframe)
        
        assert 'Age' in transformed.columns
        assert 'TotalSpend' in transformed.columns
