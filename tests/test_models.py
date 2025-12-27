"""
Unit Tests for Models Module
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    get_regression_models,
    get_classification_models,
    get_kmeans,
    get_gmm,
    find_optimal_k,
    train_baseline_models,
)
from src.config import RANDOM_STATE


class TestGetRegressionModels:
    """Tests for get_regression_models function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        models = get_regression_models()
        assert isinstance(models, dict)
    
    def test_contains_expected_models(self):
        """Test that expected models are present."""
        models = get_regression_models()
        
        expected = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest']
        for model_name in expected:
            assert model_name in models, f"Missing model: {model_name}"
    
    def test_models_can_fit(self):
        """Test that all models can fit data."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=RANDOM_STATE)
        models = get_regression_models(include_slow=False)
        
        for name, model in models.items():
            try:
                model.fit(X, y)
            except Exception as e:
                pytest.fail(f"{name} failed to fit: {e}")
    
    def test_include_slow_adds_models(self):
        """Test that include_slow=True adds additional models."""
        fast_models = get_regression_models(include_slow=False)
        slow_models = get_regression_models(include_slow=True)
        
        assert len(slow_models) > len(fast_models)


class TestGetClassificationModels:
    """Tests for get_classification_models function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        models = get_classification_models()
        assert isinstance(models, dict)
    
    def test_contains_expected_models(self):
        """Test that expected models are present."""
        models = get_classification_models()
        
        expected = ['Logistic Regression', 'Decision Tree', 'Random Forest']
        for model_name in expected:
            assert model_name in models, f"Missing model: {model_name}"
    
    def test_models_can_fit(self):
        """Test that all models can fit data."""
        X, y = make_classification(
            n_samples=100, 
            n_features=10, 
            n_classes=2,
            weights=[0.85, 0.15],
            random_state=RANDOM_STATE
        )
        models = get_classification_models(include_slow=False)
        
        for name, model in models.items():
            try:
                model.fit(X, y)
            except Exception as e:
                pytest.fail(f"{name} failed to fit: {e}")
    
    def test_class_weight_balanced(self):
        """Test that models use balanced class weights where applicable."""
        models = get_classification_models()
        
        # Logistic Regression and Random Forest should have class_weight='balanced'
        assert models['Logistic Regression'].class_weight == 'balanced'
        assert models['Random Forest'].class_weight == 'balanced'


class TestClusteringModels:
    """Tests for clustering model functions."""
    
    def test_get_kmeans_returns_model(self):
        """Test that get_kmeans returns a KMeans model."""
        from sklearn.cluster import KMeans
        
        model = get_kmeans(n_clusters=4)
        assert isinstance(model, KMeans)
        assert model.n_clusters == 4
    
    def test_get_gmm_returns_model(self):
        """Test that get_gmm returns a GaussianMixture model."""
        from sklearn.mixture import GaussianMixture
        
        model = get_gmm(n_components=3)
        assert isinstance(model, GaussianMixture)
        assert model.n_components == 3
    
    def test_kmeans_can_fit(self):
        """Test that KMeans can fit data."""
        X = np.random.randn(100, 5)
        model = get_kmeans(n_clusters=3)
        
        labels = model.fit_predict(X)
        assert len(labels) == 100
        assert set(labels).issubset({0, 1, 2})


class TestFindOptimalK:
    """Tests for find_optimal_k function."""
    
    def test_returns_expected_keys(self):
        """Test that result contains expected keys."""
        X = np.random.randn(100, 5)
        result = find_optimal_k(X, k_range=range(2, 6))
        
        expected_keys = ['k_range', 'inertias', 'silhouettes', 'best_k_silhouette']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_returns_correct_length(self):
        """Test that lists have correct length."""
        X = np.random.randn(100, 5)
        k_range = range(2, 8)
        result = find_optimal_k(X, k_range=k_range)
        
        assert len(result['k_range']) == len(k_range)
        assert len(result['inertias']) == len(k_range)
        assert len(result['silhouettes']) == len(k_range)
    
    def test_best_k_in_range(self):
        """Test that best_k is within the searched range."""
        X = np.random.randn(100, 5)
        k_range = range(2, 6)
        result = find_optimal_k(X, k_range=k_range)
        
        assert result['best_k_silhouette'] in k_range


class TestTrainBaselineModels:
    """Tests for train_baseline_models function."""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=RANDOM_STATE)
        models = {'Ridge': get_regression_models()['Ridge']}
        
        result = train_baseline_models(models, X, y, scoring='r2', cv=3)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_contains_expected_columns(self):
        """Test that result contains expected columns."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=RANDOM_STATE)
        models = {'Ridge': get_regression_models()['Ridge']}
        
        result = train_baseline_models(models, X, y, scoring='r2', cv=3)
        
        expected_cols = ['Model', 'CV_mean', 'CV_std']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_sorted_by_cv_mean(self):
        """Test that results are sorted by CV_mean descending."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=RANDOM_STATE)
        models = get_regression_models(include_slow=False)
        
        result = train_baseline_models(models, X, y, scoring='r2', cv=3)
        
        cv_means = result['CV_mean'].tolist()
        assert cv_means == sorted(cv_means, reverse=True)
