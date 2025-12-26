"""
================================================================================
Evaluation Module — Unified Metrics, Logging & Model Comparison
================================================================================

This module provides a comprehensive evaluation framework for all model types:
- Regression: R², MSE, MAE, RMSE, MAPE
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Clustering: Silhouette score, Inertia, Adjusted Rand Index

Key Components:
    1. ModelLogger: Central class for tracking all model results
    2. metric_report: Quick classification metrics summary
    3. threshold_sweep: Find optimal threshold for imbalanced classification
    4. Regression/Classification evaluation functions

Why Unified Logging?
    - Compare models across experiments consistently
    - Export results to CSV/JSON for reporting
    - Track hyperparameters and notes with each model

Usage:
    from src.evaluation import ModelLogger, metric_report, evaluate_regression
    
    logger = ModelLogger()
    logger.log_regression("Model 1", r2_train, r2_test, mse_train, mse_test)
    summary = logger.get_summary()
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Regression metrics
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    # Clustering metrics
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.model_selection import cross_val_score, learning_curve
import joblib
from datetime import datetime
import json
import os

from .config import (
    RANDOM_STATE,
    CV_FOLDS,
    MODELS_DIR,
    THRESHOLD_RANGE,
    COST_FALSE_POSITIVE,
    COST_FALSE_NEGATIVE,
)


# =============================================================================
# MODEL LOGGER CLASS
# =============================================================================
class ModelLogger:
    """
    Central logging system for tracking all model results.
    
    Supports three model types:
    - Regression: Tracks R², MSE, MAE, feature counts
    - Classification: Tracks accuracy, precision, recall, F1, AUC
    - Clustering: Tracks silhouette score, inertia, cluster counts
    
    Example
    -------
    >>> logger = ModelLogger()
    >>> logger.log_regression(
    ...     name="Linear Regression",
    ...     r2_train=0.85, r2_test=0.82,
    ...     mse_train=0.15, mse_test=0.18,
    ...     r2_cv=0.81, feature_count=20
    ... )
    >>> logger.get_summary("regression")
    """
    
    def __init__(self):
        self.regression_log = []
        self.classification_log = []
        self.clustering_log = []
    
    def _safe_round(self, value, decimals=4):
        """Safely round numeric values."""
        if value is None:
            return None
        try:
            return round(float(value), decimals)
        except (TypeError, ValueError):
            return value
    
    def log_regression(
        self,
        name,
        r2_train,
        r2_test,
        mse_train,
        mse_test,
        r2_cv=None,
        mae_train=None,
        mae_test=None,
        rmse_train=None,
        rmse_test=None,
        feature_count=None,
        notes=None,
        hyperparams=None,
    ):
        """
        Log regression model results.
        
        Parameters
        ----------
        name : str
            Model name/identifier.
        r2_train, r2_test : float
            R² scores on train and test sets.
        mse_train, mse_test : float
            Mean squared error on train and test sets.
        r2_cv : float, optional
            Cross-validated R² score.
        mae_train, mae_test : float, optional
            Mean absolute error.
        rmse_train, rmse_test : float, optional
            Root mean squared error.
        feature_count : int, optional
            Number of features used.
        notes : str, optional
            Additional notes about the model.
        hyperparams : dict, optional
            Best hyperparameters from tuning.
        """
        # Prevent duplicate entries
        if any(entry["Model"] == name for entry in self.regression_log):
            return
        
        # Calculate RMSE if not provided
        if rmse_train is None and mse_train is not None:
            rmse_train = np.sqrt(mse_train)
        if rmse_test is None and mse_test is not None:
            rmse_test = np.sqrt(mse_test)
        
        self.regression_log.append({
            "Model": name,
            "R²_train": self._safe_round(r2_train),
            "R²_test": self._safe_round(r2_test),
            "R²_CV": self._safe_round(r2_cv),
            "MSE_train": self._safe_round(mse_train),
            "MSE_test": self._safe_round(mse_test),
            "RMSE_train": self._safe_round(rmse_train),
            "RMSE_test": self._safe_round(rmse_test),
            "MAE_train": self._safe_round(mae_train),
            "MAE_test": self._safe_round(mae_test),
            "Feature_count": feature_count,
            "Notes": notes,
            "Hyperparams": hyperparams,
            "Timestamp": datetime.now().isoformat(),
        })
    
    def log_classification(
        self,
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc=None,
        pr_auc=None,
        threshold=0.5,
        cv_score=None,
        notes=None,
        hyperparams=None,
    ):
        """
        Log classification model results.
        
        Parameters
        ----------
        name : str
            Model name/identifier.
        accuracy, precision, recall, f1 : float
            Standard classification metrics.
        roc_auc : float, optional
            Area under ROC curve.
        pr_auc : float, optional
            Area under Precision-Recall curve.
        threshold : float, default=0.5
            Decision threshold used.
        cv_score : float, optional
            Cross-validated score.
        notes : str, optional
            Additional notes.
        hyperparams : dict, optional
            Best hyperparameters.
        """
        if any(entry["Model"] == name for entry in self.classification_log):
            return
        
        self.classification_log.append({
            "Model": name,
            "Accuracy": self._safe_round(accuracy),
            "Precision": self._safe_round(precision),
            "Recall": self._safe_round(recall),
            "F1": self._safe_round(f1),
            "ROC_AUC": self._safe_round(roc_auc),
            "PR_AUC": self._safe_round(pr_auc),
            "Threshold": threshold,
            "CV_Score": self._safe_round(cv_score),
            "Notes": notes,
            "Hyperparams": hyperparams,
            "Timestamp": datetime.now().isoformat(),
        })
    
    def log_clustering(
        self,
        name,
        n_clusters,
        silhouette,
        inertia=None,
        ari=None,
        notes=None,
    ):
        """
        Log clustering model results.
        
        Parameters
        ----------
        name : str
            Model name/identifier.
        n_clusters : int
            Number of clusters.
        silhouette : float
            Silhouette score (-1 to 1, higher is better).
        inertia : float, optional
            Sum of squared distances to centroids (K-Means).
        ari : float, optional
            Adjusted Rand Index (if ground truth available).
        notes : str, optional
            Additional notes.
        """
        if any(entry["Model"] == name for entry in self.clustering_log):
            return
        
        self.clustering_log.append({
            "Model": name,
            "N_Clusters": n_clusters,
            "Silhouette": self._safe_round(silhouette),
            "Inertia": self._safe_round(inertia),
            "ARI": self._safe_round(ari),
            "Notes": notes,
            "Timestamp": datetime.now().isoformat(),
        })
    
    def get_summary(self, model_type="regression", sort_by=None, ascending=False):
        """
        Get summary DataFrame of logged models.
        
        Parameters
        ----------
        model_type : str
            One of "regression", "classification", "clustering".
        sort_by : str, optional
            Column to sort by (e.g., "R²_test", "F1").
        ascending : bool, default=False
            Sort order.
        
        Returns
        -------
        pd.DataFrame
            Summary of model results.
        """
        log_map = {
            "regression": self.regression_log,
            "classification": self.classification_log,
            "clustering": self.clustering_log,
        }
        
        log = log_map.get(model_type, [])
        
        if not log:
            return pd.DataFrame()
        
        df = pd.DataFrame(log)
        
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        
        return df
    
    def export_to_csv(self, filepath, model_type="regression"):
        """Export model log to CSV file."""
        df = self.get_summary(model_type)
        df.to_csv(filepath, index=False)
        print(f"✅ Exported {len(df)} {model_type} models to {filepath}")
    
    def export_to_json(self, filepath, model_type="regression"):
        """Export model log to JSON file."""
        log_map = {
            "regression": self.regression_log,
            "classification": self.classification_log,
            "clustering": self.clustering_log,
        }
        
        with open(filepath, "w") as f:
            json.dump(log_map.get(model_type, []), f, indent=2, default=str)
        
        print(f"✅ Exported {model_type} models to {filepath}")
    
    def get_best_model(self, model_type="regression", metric="R²_test"):
        """Get the best model based on specified metric."""
        df = self.get_summary(model_type)
        
        if df.empty or metric not in df.columns:
            return None
        
        best_idx = df[metric].idxmax()
        return df.loc[best_idx].to_dict()


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================
def metric_report(y_true, y_pred, y_proba=None, label="model"):
    """
    Compute comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities for positive class.
    label : str
        Model identifier.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    
    Example
    -------
    >>> metrics = metric_report(y_test, y_pred, y_proba, "Random Forest")
    >>> print(metrics)
    """
    result = {
        "model": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
        result["pr_auc"] = average_precision_score(y_true, y_proba)
    
    return result


def threshold_sweep(y_true, y_score, thresholds=None):
    """
    Sweep thresholds to analyze precision/recall/F1 trade-offs.
    
    This is essential for imbalanced classification where the default
    threshold of 0.5 may not be optimal.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities for positive class.
    thresholds : array-like, optional
        Thresholds to evaluate.
    
    Returns
    -------
    pd.DataFrame
        Metrics at each threshold.
    
    Example
    -------
    >>> sweep_df = threshold_sweep(y_test, y_proba)
    >>> best_f1_threshold = sweep_df.loc[sweep_df['f1'].idxmax(), 'threshold']
    """
    if thresholds is None:
        thresholds = THRESHOLD_RANGE
    
    results = []
    
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        })
    
    return pd.DataFrame(results)


def choose_threshold_by_objective(sweep_df, objective="f1_max", target=0.8):
    """
    Choose optimal threshold based on business objective.
    
    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output from threshold_sweep().
    objective : str
        One of:
        - "f1_max": Maximize F1 score
        - "precision_at": Achieve target precision, maximize recall
        - "recall_at": Achieve target recall, maximize precision
    target : float
        Target value for precision_at or recall_at objectives.
    
    Returns
    -------
    tuple
        (threshold, description)
    
    Example
    -------
    >>> t, desc = choose_threshold_by_objective(sweep_df, "precision_at", 0.7)
    >>> print(f"Use threshold {t:.2f}: {desc}")
    """
    if objective == "f1_max":
        idx = sweep_df["f1"].idxmax()
        return float(sweep_df.loc[idx, "threshold"]), "F1-maximizing"
    
    if objective == "precision_at":
        df = sweep_df[sweep_df["precision"] >= target]
        if len(df) == 0:
            return float(sweep_df.iloc[-1]["threshold"]), f"precision≥{target} (best available)"
        idx = df["recall"].idxmax()
        return float(df.loc[idx, "threshold"]), f"precision≥{target}, max recall"
    
    if objective == "recall_at":
        df = sweep_df[sweep_df["recall"] >= target]
        if len(df) == 0:
            return float(sweep_df.iloc[0]["threshold"]), f"recall≥{target} (best available)"
        idx = df["precision"].idxmax()
        return float(df.loc[idx, "threshold"]), f"recall≥{target}, max precision"
    
    raise ValueError(f"Unknown objective: {objective}")


def expected_cost(y_true, y_score, threshold, c_fp=None, c_fn=None):
    """
    Calculate expected cost for asymmetric misclassification costs.
    
    In marketing campaigns:
    - False Positive (FP): Contact someone who won't respond → wasted resources
    - False Negative (FN): Miss a potential customer → lost opportunity (often more costly)
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted probabilities.
    threshold : float
        Decision threshold.
    c_fp : float, optional
        Cost of false positive. Default from config.
    c_fn : float, optional
        Cost of false negative. Default from config.
    
    Returns
    -------
    float
        Total expected cost.
    """
    if c_fp is None:
        c_fp = COST_FALSE_POSITIVE
    if c_fn is None:
        c_fn = COST_FALSE_NEGATIVE
    
    y_pred = (y_score >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return fp * c_fp + fn * c_fn


def find_cost_optimal_threshold(y_true, y_score, c_fp=None, c_fn=None, thresholds=None):
    """
    Find threshold that minimizes total misclassification cost.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted probabilities.
    c_fp, c_fn : float, optional
        Costs of false positive/negative.
    thresholds : array-like, optional
        Thresholds to evaluate.
    
    Returns
    -------
    tuple
        (optimal_threshold, costs_array, thresholds_array)
    """
    if thresholds is None:
        thresholds = THRESHOLD_RANGE
    
    costs = [expected_cost(y_true, y_score, t, c_fp, c_fn) for t in thresholds]
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, costs, thresholds


# =============================================================================
# REGRESSION EVALUATION
# =============================================================================
def evaluate_regression(model, X_train, X_test, y_train, y_test, cv=None):
    """
    Comprehensive regression model evaluation.
    
    Parameters
    ----------
    model : estimator
        Fitted sklearn model or pipeline.
    X_train, X_test : array-like
        Feature matrices.
    y_train, y_test : array-like
        Target arrays.
    cv : int or cross-validator, optional
        Cross-validation strategy.
    
    Returns
    -------
    dict
        Dictionary of all regression metrics.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Core metrics
    metrics = {
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "mse_train": mean_squared_error(y_train, y_train_pred),
        "mse_test": mean_squared_error(y_test, y_test_pred),
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "mae_test": mean_absolute_error(y_test, y_test_pred),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }
    
    # MAPE (watch out for zeros in target)
    try:
        metrics["mape_train"] = mean_absolute_percentage_error(y_train, y_train_pred)
        metrics["mape_test"] = mean_absolute_percentage_error(y_test, y_test_pred)
    except:
        metrics["mape_train"] = None
        metrics["mape_test"] = None
    
    # Cross-validation
    if cv is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        metrics["r2_cv_mean"] = cv_scores.mean()
        metrics["r2_cv_std"] = cv_scores.std()
    
    # Store predictions for plotting
    metrics["y_train_pred"] = y_train_pred
    metrics["y_test_pred"] = y_test_pred
    
    return metrics


def evaluate_classification(model, X_train, X_test, y_train, y_test, threshold=0.5, cv=None):
    """
    Comprehensive classification model evaluation.
    
    Parameters
    ----------
    model : estimator
        Fitted sklearn classifier.
    X_train, X_test : array-like
        Feature matrices.
    y_train, y_test : array-like
        Target arrays.
    threshold : float, default=0.5
        Decision threshold.
    cv : int or cross-validator, optional
        Cross-validation strategy.
    
    Returns
    -------
    dict
        Dictionary of all classification metrics.
    """
    # Get probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Apply threshold
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    metrics = {
        "threshold": threshold,
        # Train metrics
        "accuracy_train": accuracy_score(y_train, y_train_pred),
        "precision_train": precision_score(y_train, y_train_pred, zero_division=0),
        "recall_train": recall_score(y_train, y_train_pred, zero_division=0),
        "f1_train": f1_score(y_train, y_train_pred, zero_division=0),
        "roc_auc_train": roc_auc_score(y_train, y_train_proba),
        # Test metrics
        "accuracy_test": accuracy_score(y_test, y_test_pred),
        "precision_test": precision_score(y_test, y_test_pred, zero_division=0),
        "recall_test": recall_score(y_test, y_test_pred, zero_division=0),
        "f1_test": f1_score(y_test, y_test_pred, zero_division=0),
        "roc_auc_test": roc_auc_score(y_test, y_test_proba),
        "pr_auc_test": average_precision_score(y_test, y_test_proba),
    }
    
    # Cross-validation
    if cv is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        metrics["roc_auc_cv_mean"] = cv_scores.mean()
        metrics["roc_auc_cv_std"] = cv_scores.std()
    
    # Store for plotting
    metrics["y_test_pred"] = y_test_pred
    metrics["y_test_proba"] = y_test_proba
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_test_pred)
    
    return metrics


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================
def save_model(model, name, directory=None):
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : estimator
        Trained sklearn model or pipeline.
    name : str
        Model name (without extension).
    directory : str, optional
        Directory to save to. Default from config.
    
    Returns
    -------
    str
        Path to saved model.
    """
    if directory is None:
        directory = MODELS_DIR
    
    os.makedirs(directory, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pkl"
    filepath = os.path.join(directory, filename)
    
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")
    
    return filepath


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to saved model file.
    
    Returns
    -------
    estimator
        Loaded model.
    """
    model = joblib.load(filepath)
    print(f"✅ Model loaded from {filepath}")
    return model


# =============================================================================
# LEARNING CURVES
# =============================================================================
def compute_learning_curves(model, X, y, cv=5, train_sizes=None, scoring="r2"):
    """
    Compute learning curves to diagnose bias/variance.
    
    Parameters
    ----------
    model : estimator
        Sklearn model or pipeline.
    X : array-like
        Feature matrix.
    y : array-like
        Target array.
    cv : int
        Cross-validation folds.
    train_sizes : array-like, optional
        Training set sizes to evaluate.
    scoring : str
        Scoring metric.
    
    Returns
    -------
    dict
        Dictionary with train_sizes, train_scores, test_scores.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    
    return {
        "train_sizes": train_sizes_abs,
        "train_scores_mean": train_scores.mean(axis=1),
        "train_scores_std": train_scores.std(axis=1),
        "test_scores_mean": test_scores.mean(axis=1),
        "test_scores_std": test_scores.std(axis=1),
    }
