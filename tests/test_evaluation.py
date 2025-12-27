"""
Unit Tests for Evaluation Module
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    ModelLogger,
    evaluate_regression,
    evaluate_classification,
)
from src.config import RANDOM_STATE


class TestModelLogger:
    """Tests for ModelLogger class."""
    
    def test_init_creates_empty_logs(self):
        """Test that ModelLogger initializes with empty logs."""
        logger = ModelLogger()
        
        assert logger.regression_log == []
        assert logger.classification_log == []
        assert logger.clustering_log == []
    
    def test_log_regression(self):
        """Test logging regression results."""
        logger = ModelLogger()
        
        logger.log_regression(
            name="Test Model",
            r2_train=0.95,
            r2_test=0.90,
            mse_train=0.05,
            mse_test=0.10
        )
        
        assert len(logger.regression_log) == 1
        assert logger.regression_log[0]['Model'] == "Test Model"
        assert logger.regression_log[0]['R²_train'] == 0.95
    
    def test_log_regression_prevents_duplicates(self):
        """Test that duplicate model names are not logged."""
        logger = ModelLogger()
        
        logger.log_regression("Model1", 0.9, 0.85, 0.1, 0.15)
        logger.log_regression("Model1", 0.95, 0.88, 0.05, 0.12)  # Same name
        
        assert len(logger.regression_log) == 1
    
    def test_log_classification(self):
        """Test logging classification results."""
        logger = ModelLogger()
        
        logger.log_classification(
            name="Test Classifier",
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            roc_auc=0.88
        )
        
        assert len(logger.classification_log) == 1
        assert logger.classification_log[0]['Model'] == "Test Classifier"
        assert logger.classification_log[0]['ROC_AUC'] == 0.88
    
    def test_log_clustering(self):
        """Test logging clustering results."""
        logger = ModelLogger()
        
        logger.log_clustering(
            name="K-Means",
            n_clusters=4,
            silhouette=0.45,
            inertia=1500
        )
        
        assert len(logger.clustering_log) == 1
        assert logger.clustering_log[0]['Model'] == "K-Means"
        assert logger.clustering_log[0]['Silhouette'] == 0.45
    
    def test_safe_round_handles_none(self):
        """Test that _safe_round handles None values."""
        logger = ModelLogger()
        
        result = logger._safe_round(None)
        assert result is None
    
    def test_safe_round_rounds_floats(self):
        """Test that _safe_round rounds float values."""
        logger = ModelLogger()
        
        result = logger._safe_round(0.123456789, decimals=4)
        assert result == 0.1235


class TestEvaluateRegression:
    """Tests for evaluate_regression function."""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression test data."""
        X, y = make_regression(
            n_samples=200, 
            n_features=10, 
            noise=10,
            random_state=RANDOM_STATE
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def test_returns_dict(self, regression_data):
        """Test that evaluate_regression returns a dictionary."""
        result = evaluate_regression(
            regression_data['model'],
            regression_data['X_train'],
            regression_data['X_test'],
            regression_data['y_train'],
            regression_data['y_test']
        )
        
        assert isinstance(result, dict)
    
    def test_contains_expected_metrics(self, regression_data):
        """Test that result contains expected metrics."""
        result = evaluate_regression(
            regression_data['model'],
            regression_data['X_train'],
            regression_data['X_test'],
            regression_data['y_train'],
            regression_data['y_test']
        )
        
        expected = ['r2_train', 'r2_test', 'mse_train', 'mse_test']
        for metric in expected:
            assert metric in result, f"Missing metric: {metric}"
    
    def test_r2_in_valid_range(self, regression_data):
        """Test that R² values are in valid range."""
        result = evaluate_regression(
            regression_data['model'],
            regression_data['X_train'],
            regression_data['X_test'],
            regression_data['y_train'],
            regression_data['y_test']
        )
        
        # R² can be negative for bad models, but typically between -1 and 1
        assert result['r2_train'] <= 1.0
        assert result['r2_test'] <= 1.0


class TestEvaluateClassification:
    """Tests for evaluate_classification function."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            weights=[0.85, 0.15],
            random_state=RANDOM_STATE
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def test_returns_dict(self, classification_data):
        """Test that evaluate_classification returns a dictionary."""
        result = evaluate_classification(
            classification_data['model'],
            classification_data['X_test'],
            classification_data['y_test']
        )
        
        assert isinstance(result, dict)
    
    def test_contains_expected_metrics(self, classification_data):
        """Test that result contains expected metrics."""
        result = evaluate_classification(
            classification_data['model'],
            classification_data['X_test'],
            classification_data['y_test']
        )
        
        expected = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected:
            assert metric in result, f"Missing metric: {metric}"
    
    def test_metrics_in_valid_range(self, classification_data):
        """Test that metrics are in valid range [0, 1]."""
        result = evaluate_classification(
            classification_data['model'],
            classification_data['X_test'],
            classification_data['y_test']
        )
        
        for metric_name, value in result.items():
            if isinstance(value, float):
                assert 0 <= value <= 1, f"{metric_name} out of range: {value}"
