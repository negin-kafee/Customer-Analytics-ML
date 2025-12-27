"""
================================================================================
Models Module — Model Factory and Training Utilities
================================================================================

This module provides:
- Factory functions for regression, classification, and clustering models
- Hyperparameter grids for tuning
- Unified training and evaluation interface

Design Principles:
    1. Consistent API across all model types
    2. Easy model comparison with standardized output
    3. Support for hyperparameter tuning via GridSearchCV
    4. Clear separation of concerns

Usage:
    from src.models import get_regression_models, train_with_gridsearch
    
    models = get_regression_models()
    best_model, results = train_with_gridsearch(models['ridge'], X, y, param_grid)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    make_scorer, r2_score, mean_squared_error,
    f1_score, roc_auc_score, silhouette_score
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .config import (
    RANDOM_STATE, CV_FOLDS,
    TREE_PARAM_GRID, RF_PARAM_GRID, XGB_PARAM_GRID
)


# =============================================================================
# REGRESSION MODELS
# =============================================================================
def get_regression_models(include_slow: bool = False) -> Dict[str, Any]:
    """
    Get dictionary of regression models.
    
    Parameters
    ----------
    include_slow : bool
        Include computationally expensive models (SVR, Neural Network).
    
    Returns
    -------
    dict
        Model name -> unfitted model instance.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(
            random_state=RANDOM_STATE,
            estimator=DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)
        ),
        "KNN": KNeighborsRegressor(n_jobs=-1),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
    
    if include_slow:
        models["SVR"] = SVR()
        models["MLP"] = MLPRegressor(
            random_state=RANDOM_STATE,
            max_iter=500,
            early_stopping=True
        )
    
    return models


def get_regression_param_grids() -> Dict[str, Dict]:
    """
    Get hyperparameter grids for regression models.
    
    Returns
    -------
    dict
        Model name -> parameter grid.
    """
    grids = {
        "Ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        "Lasso": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
        },
        "ElasticNet": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.25, 0.5, 0.75]
        },
        "Decision Tree": TREE_PARAM_GRID,
        "Random Forest": RF_PARAM_GRID,
        "Gradient Boosting": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        },
    }
    
    if HAS_XGBOOST:
        grids["XGBoost"] = XGB_PARAM_GRID
    
    return grids


# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================
def get_classification_models(include_slow: bool = False) -> Dict[str, Any]:
    """
    Get dictionary of classification models.
    
    Parameters
    ----------
    include_slow : bool
        Include computationally expensive models (SVC, Neural Network).
    
    Returns
    -------
    dict
        Model name -> unfitted model instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            algorithm="SAMME",
            random_state=RANDOM_STATE,
            estimator=DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
        ),
        "KNN": KNeighborsClassifier(n_jobs=-1),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
            scale_pos_weight=6  # Approximate ratio for imbalanced data
        )
    
    if include_slow:
        models["SVC"] = SVC(
            random_state=RANDOM_STATE,
            probability=True,
            class_weight="balanced"
        )
        models["MLP"] = MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=500,
            early_stopping=True
        )
    
    return models


def get_classification_param_grids() -> Dict[str, Dict]:
    """
    Get hyperparameter grids for classification models.
    
    Returns
    -------
    dict
        Model name -> parameter grid.
    """
    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"]
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        },
    }
    
    if HAS_XGBOOST:
        grids["XGBoost"] = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }
    
    return grids


# =============================================================================
# CLUSTERING MODELS
# =============================================================================
def get_kmeans(n_clusters: int = 4) -> KMeans:
    """Get configured KMeans model."""
    return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)


def get_gmm(n_components: int = 4) -> GaussianMixture:
    """Get configured Gaussian Mixture Model."""
    return GaussianMixture(
        n_components=n_components,
        random_state=RANDOM_STATE,
        covariance_type="full",
        n_init=5
    )


def get_hierarchical(n_clusters: int = 4, linkage: str = "ward") -> AgglomerativeClustering:
    """Get configured Agglomerative Clustering model."""
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )


def get_dbscan(eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
    """Get configured DBSCAN model."""
    return DBSCAN(eps=eps, min_samples=min_samples)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================
def train_with_gridsearch(
    model,
    X_train,
    y_train,
    param_grid: Dict,
    scoring: str = "r2",
    cv: int = CV_FOLDS,
    n_jobs: int = -1,
    verbose: int = 0
) -> Tuple[Any, Dict]:
    """
    Train model with GridSearchCV hyperparameter tuning.
    
    Parameters
    ----------
    model : estimator
        Sklearn-compatible model.
    X_train : array-like
        Training features.
    y_train : array-like
        Training targets.
    param_grid : dict
        Hyperparameter search space.
    scoring : str
        Scoring metric for CV.
    cv : int
        Number of cross-validation folds.
    n_jobs : int
        Parallel jobs (-1 for all cores).
    verbose : int
        Verbosity level.
    
    Returns
    -------
    tuple
        (best_model, results_dict)
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    results = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": pd.DataFrame(grid_search.cv_results_)
    }
    
    return grid_search.best_estimator_, results


def train_baseline_models(
    models: Dict[str, Any],
    X_train,
    y_train,
    scoring: str = "r2",
    cv: int = CV_FOLDS
) -> pd.DataFrame:
    """
    Train multiple models and get CV scores (no hyperparameter tuning).
    
    Parameters
    ----------
    models : dict
        Model name -> unfitted model.
    X_train : array-like
        Training features.
    y_train : array-like
        Training targets.
    scoring : str
        Scoring metric.
    cv : int
        Cross-validation folds.
    
    Returns
    -------
    pd.DataFrame
        Model comparison results.
    """
    results = []
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            results.append({
                "Model": name,
                "CV_mean": scores.mean(),
                "CV_std": scores.std(),
                "CV_min": scores.min(),
                "CV_max": scores.max()
            })
            print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"{name}: FAILED - {e}")
    
    return pd.DataFrame(results).sort_values("CV_mean", ascending=False)


def find_optimal_k(
    X,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow and silhouette methods.
    
    Parameters
    ----------
    X : array-like
        Feature matrix (preferably scaled).
    k_range : range
        Range of k values to test.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    dict
        Results with inertias, silhouettes, and best_k.
    """
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    # Find best k by silhouette
    best_k = k_range[np.argmax(silhouettes)]
    
    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k_silhouette": best_k
    }


def compute_cluster_profiles(
    data: pd.DataFrame,
    labels,
    features: List[str]
) -> pd.DataFrame:
    """
    Compute mean feature values for each cluster.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data with features.
    labels : array-like
        Cluster assignments.
    features : list
        Feature columns to profile.
    
    Returns
    -------
    pd.DataFrame
        Mean values per cluster.
    """
    data_copy = data.copy()
    data_copy["Cluster"] = labels
    
    profile = data_copy.groupby("Cluster")[features].mean()
    return profile


def compute_spending_mix(
    data: pd.DataFrame,
    labels,
    spending_cols: List[str]
) -> pd.DataFrame:
    """
    Compute share of wallet (spending mix) per cluster.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data with spending columns.
    labels : array-like
        Cluster assignments.
    spending_cols : list
        Columns representing spending categories.
    
    Returns
    -------
    pd.DataFrame
        Share of spending per category per cluster.
    """
    data_copy = data.copy()
    data_copy["Cluster"] = labels
    
    # Mean spending per cluster
    cluster_spend = data_copy.groupby("Cluster")[spending_cols].mean()
    
    # Normalize to get share of wallet
    totals = cluster_spend.sum(axis=1)
    mix = cluster_spend.div(totals, axis=0)
    
    return mix


# =============================================================================
# MODEL EXPLANATION
# =============================================================================
def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models.
    
    Parameters
    ----------
    model : fitted model
        Model with feature_importances_ attribute.
    feature_names : list
        Feature names.
    
    Returns
    -------
    pd.DataFrame
        Sorted feature importances.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    return df


def get_coefficients(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract coefficients from linear models.
    
    Parameters
    ----------
    model : fitted model
        Model with coef_ attribute.
    feature_names : list
        Feature names.
    
    Returns
    -------
    pd.DataFrame
        Sorted coefficients by absolute value.
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not have coef_ attribute")
    
    coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
    
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)
    
    return df


# =============================================================================
# DEEP LEARNING MODEL BUILDERS (TensorFlow/Keras)
# =============================================================================
def build_mlp_regressor(
    input_dim: int,
    hidden_layers: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
):
    """
    Build MLP for regression using Keras.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : list
        List of units per hidden layer.
    dropout_rate : float
        Dropout rate after each hidden layer.
    learning_rate : float
        Adam optimizer learning rate.
    
    Returns
    -------
    keras.Model
        Compiled MLP model.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1))  # Regression output
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    
    return model


def build_mlp_classifier(
    input_dim: int,
    hidden_layers: List[int] = [64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
):
    """
    Build MLP for binary classification using Keras.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : list
        List of units per hidden layer.
    dropout_rate : float
        Dropout rate after each hidden layer.
    learning_rate : float
        Adam optimizer learning rate.
    
    Returns
    -------
    keras.Model
        Compiled MLP classifier.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation="sigmoid"))  # Binary classification
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    
    return model


def get_early_stopping(patience: int = 10, restore_best: bool = True):
    """
    Get early stopping callback for Keras training.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping.
    restore_best : bool
        Whether to restore best weights.
    
    Returns
    -------
    keras.callbacks.EarlyStopping
        Configured callback.
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError("TensorFlow is required.")
    
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=restore_best,
        verbose=1
    )
