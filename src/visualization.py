"""
================================================================================
Visualization Module — Consistent, Publication-Ready Plots
================================================================================

This module provides a comprehensive visualization toolkit for ML analysis:
- EDA: Distributions, correlations, boxplots, scatterplots
- Regression: Residual plots, actual vs predicted, learning curves
- Classification: ROC/PR curves, confusion matrices, threshold analysis
- Clustering: Elbow plots, silhouette analysis, PCA scatter, dendrograms

Design Principles:
    1. Consistent color scheme (purple primary, teal secondary)
    2. Informative titles and labels
    3. Proper figure sizing for notebooks
    4. Export-ready (high DPI, clean formatting)

Usage:
    from src.visualization import plot_distribution, plot_correlation_matrix
    
    plot_distribution(df, 'Income', title='Income Distribution')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    ConfusionMatrixDisplay, average_precision_score
)
from sklearn.decomposition import PCA

from .config import (
    MAIN_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
    FIGURE_SIZE_SMALL,
    FIGURE_SIZE_MEDIUM,
    FIGURE_SIZE_LARGE,
    FIGURE_SIZE_WIDE,
    FIGURE_DPI,
)


# =============================================================================
# STYLE SETUP
# =============================================================================
def set_style():
    """Set consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.figsize": FIGURE_SIZE_MEDIUM,
        "figure.dpi": FIGURE_DPI,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_palette([MAIN_COLOR, SECONDARY_COLOR, ACCENT_COLOR])


# Custom colormap for correlations
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom_purple",
    ["#FFFFFF", MAIN_COLOR]
)

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "custom_diverging",
    [NEGATIVE_COLOR, "#FFFFFF", POSITIVE_COLOR]
)


# =============================================================================
# EDA VISUALIZATIONS
# =============================================================================
def plot_distribution(data, column, title=None, bins=40, kde=True, figsize=None):
    """
    Plot histogram with optional KDE for a single column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to plot.
    title : str, optional
        Plot title.
    bins : int
        Number of histogram bins.
    kde : bool
        Whether to overlay KDE curve.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_SMALL
    plt.figure(figsize=figsize)
    
    sns.histplot(data[column], bins=bins, kde=kde, color=MAIN_COLOR, alpha=0.7)
    
    title = title or f"Distribution of {column}"
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_distributions_grid(data, columns, ncols=3, bins=30, figsize=None):
    """
    Plot distribution grid for multiple columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    columns : list
        Column names to plot.
    ncols : int
        Number of columns in grid.
    bins : int
        Number of histogram bins.
    figsize : tuple, optional
        Figure size.
    """
    nrows = int(np.ceil(len(columns) / ncols))
    figsize = figsize or (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    for i, col in enumerate(columns):
        sns.histplot(data[col], bins=bins, kde=True, color=MAIN_COLOR, ax=axes[i])
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Feature Distributions", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_boxplots_grid(data, columns, ncols=3, figsize=None):
    """
    Plot boxplot grid for outlier visualization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    columns : list
        Column names to plot.
    ncols : int
        Number of columns in grid.
    figsize : tuple, optional
        Figure size.
    """
    nrows = int(np.ceil(len(columns) / ncols))
    figsize = figsize or (5 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    for i, col in enumerate(columns):
        sns.boxplot(x=data[col], color=MAIN_COLOR, ax=axes[i])
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Feature Boxplots (Outlier Detection)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_countplot(data, column, title=None, figsize=None, order_by_count=True):
    """
    Plot bar chart for categorical variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    column : str
        Categorical column name.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    order_by_count : bool
        Whether to order bars by frequency.
    """
    figsize = figsize or FIGURE_SIZE_SMALL
    plt.figure(figsize=figsize)
    
    order = data[column].value_counts().index if order_by_count else None
    sns.countplot(x=data[column], order=order, color=MAIN_COLOR)
    
    title = title or f"Distribution of {column}"
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data, columns=None, method="spearman", figsize=None, annot=True):
    """
    Plot correlation heatmap.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    columns : list, optional
        Columns to include. If None, uses all numeric columns.
    method : str
        Correlation method: "pearson" or "spearman".
    figsize : tuple, optional
        Figure size.
    annot : bool
        Whether to annotate cells with values.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    corr = data[columns].corr(method=method)
    
    figsize = figsize or (len(columns) * 0.7, len(columns) * 0.6)
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        corr,
        annot=annot,
        cmap=DIVERGING_CMAP,
        center=0,
        linewidths=0.5,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    
    plt.title(f"Correlation Matrix ({method.capitalize()})", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


def plot_target_correlation(data, target, top_n=15, method="spearman"):
    """
    Plot top correlations with target variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    target : str
        Target column name.
    top_n : int
        Number of top correlations to show.
    method : str
        Correlation method.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if target not in numeric_cols:
        print(f"Target '{target}' not found in numeric columns.")
        return
    
    corr = data[numeric_cols].corr(method=method)[target].drop(target)
    corr_sorted = corr.abs().sort_values(ascending=False).head(top_n)
    corr_values = corr.loc[corr_sorted.index]
    
    plt.figure(figsize=(8, 6))
    colors = [POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR for v in corr_values]
    corr_values.plot(kind="barh", color=colors)
    
    plt.axvline(0, color="black", linewidth=0.5)
    plt.title(f"Top {top_n} Correlations with {target}", fontsize=14)
    plt.xlabel(f"{method.capitalize()} Correlation")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_scatter(data, x, y, hue=None, title=None, figsize=None, alpha=0.5):
    """
    Plot scatter plot for two variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    x, y : str
        Column names for x and y axes.
    hue : str, optional
        Column for color coding.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    alpha : float
        Point transparency.
    """
    figsize = figsize or FIGURE_SIZE_SMALL
    plt.figure(figsize=figsize)
    
    sns.scatterplot(data=data, x=x, y=y, hue=hue, alpha=alpha, color=MAIN_COLOR if hue is None else None)
    
    title = title or f"{y} vs {x}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_category(data, x, y, title=None, figsize=None):
    """
    Plot boxplot of numeric variable by categorical groups.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    x : str
        Categorical column.
    y : str
        Numeric column.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_MEDIUM
    plt.figure(figsize=figsize)
    
    sns.boxplot(data=data, x=x, y=y, color=MAIN_COLOR)
    
    title = title or f"{y} by {x}"
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# =============================================================================
# ANOMALY DETECTION VISUALIZATIONS
# =============================================================================
def plot_anomaly_scatter_pca(X, anomaly_mask, title="Anomaly Detection (PCA 2D)", figsize=None):
    """
    Plot PCA 2D scatter with anomalies highlighted.
    
    Parameters
    ----------
    X : array-like
        Feature matrix (preferably scaled).
    anomaly_mask : array-like
        Boolean array indicating anomalies (True = anomaly).
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_MEDIUM
    
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=figsize)
    
    # Plot normal points
    plt.scatter(
        X_pca[~anomaly_mask, 0],
        X_pca[~anomaly_mask, 1],
        s=20, alpha=0.5, label="Normal", color=SECONDARY_COLOR
    )
    
    # Plot anomalies
    plt.scatter(
        X_pca[anomaly_mask, 0],
        X_pca[anomaly_mask, 1],
        s=40, alpha=0.8, label="Anomaly", color=NEGATIVE_COLOR
    )
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_outlier_summary(outlier_df, title="IQR Outlier Detection Summary", figsize=None):
    """
    Plot horizontal bar chart of outlier percentages by feature.
    
    Parameters
    ----------
    outlier_df : pd.DataFrame
        Output from detect_outliers_iqr() with columns: feature, pct_outliers.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or (8, 6)
    
    df = outlier_df.sort_values("pct_outliers", ascending=True)
    
    plt.figure(figsize=figsize)
    plt.barh(df["feature"], df["pct_outliers"], color=MAIN_COLOR)
    plt.xlabel("% Outliers")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# REGRESSION VISUALIZATIONS
# =============================================================================
def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", figsize=None):
    """
    Plot actual vs predicted values for regression.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_SMALL
    
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, color=MAIN_COLOR)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction")
    
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{title} — Scatter")
    axes[0].legend()
    
    # KDE comparison
    sns.kdeplot(y_true, label="Actual", fill=True, alpha=0.3, color=MAIN_COLOR, ax=axes[1])
    sns.kdeplot(y_pred, label="Predicted", fill=True, alpha=0.3, color=SECONDARY_COLOR, ax=axes[1])
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"{title} — Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Analysis", figsize=None):
    """
    Plot residual diagnostics for regression.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_WIDE
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color=MAIN_COLOR)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")
    
    # Residual distribution
    sns.histplot(residuals, kde=True, color=MAIN_COLOR, ax=axes[1])
    axes[1].set_xlabel("Residuals")
    axes[1].set_title("Residual Distribution")
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(learning_curve_data, title="Learning Curves", figsize=None):
    """
    Plot learning curves to diagnose bias/variance.
    
    Parameters
    ----------
    learning_curve_data : dict
        Output from compute_learning_curves().
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_MEDIUM
    
    train_sizes = learning_curve_data["train_sizes"]
    train_mean = learning_curve_data["train_scores_mean"]
    train_std = learning_curve_data["train_scores_std"]
    test_mean = learning_curve_data["test_scores_mean"]
    test_std = learning_curve_data["test_scores_std"]
    
    plt.figure(figsize=figsize)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=MAIN_COLOR)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color=SECONDARY_COLOR)
    
    plt.plot(train_sizes, train_mean, "o-", color=MAIN_COLOR, label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color=SECONDARY_COLOR, label="Cross-validation score")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, title="Feature Importance", top_n=20, figsize=None):
    """
    Plot horizontal bar chart of feature importances.
    
    Parameters
    ----------
    feature_names : array-like
        Feature names.
    importances : array-like
        Importance values.
    title : str
        Plot title.
    top_n : int
        Number of top features to show.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or (8, 10)
    
    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(indices)), importances[indices], color=MAIN_COLOR)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_coefficient_path(alphas, coefs, feature_names, top_n=10, title="Lasso Coefficient Path", figsize=None):
    """
    Plot Lasso coefficient path across regularization strengths.
    
    Parameters
    ----------
    alphas : array-like
        Alpha values.
    coefs : array-like
        Coefficient matrix (n_alphas, n_features).
    feature_names : array-like
        Feature names.
    top_n : int
        Number of top features to highlight.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or (10, 6)
    
    # Find top features at middle alpha
    mid_idx = len(alphas) // 2
    top_idx = np.argsort(np.abs(coefs[mid_idx]))[-top_n:]
    
    plt.figure(figsize=figsize)
    
    for i in top_idx:
        plt.plot(np.log10(alphas), coefs[:, i], linewidth=2, label=feature_names[i])
    
    plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
    plt.xlabel("log10(alpha)")
    plt.ylabel("Coefficient Value")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(summary_df, metric="R²_test", title="Model Comparison", figsize=None):
    """
    Plot horizontal bar chart comparing models.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Model summary from ModelLogger.get_summary().
    metric : str
        Metric column to compare.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or (10, max(6, len(summary_df) * 0.4))
    
    df = summary_df.sort_values(metric, ascending=True)
    
    plt.figure(figsize=figsize)
    plt.barh(df["Model"], df[metric], color=MAIN_COLOR)
    plt.xlabel(metric)
    plt.title(f"{title} — {metric}")
    plt.tight_layout()
    plt.show()


# =============================================================================
# CLASSIFICATION VISUALIZATIONS
# =============================================================================
def plot_roc_pr_curves(y_true, y_score, title_prefix="", figsize=None):
    """
    Plot ROC and Precision-Recall curves side by side.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities.
    title_prefix : str
        Prefix for titles.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_WIDE
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color=MAIN_COLOR, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{title_prefix} ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    baseline = y_true.mean()
    
    axes[1].plot(recall, precision, color=MAIN_COLOR, linewidth=2, label=f"AP = {pr_auc:.3f}")
    axes[1].axhline(baseline, color="k", linestyle="--", label=f"Baseline = {baseline:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", figsize=None):
    """
    Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_SMALL
    
    plt.figure(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=CUSTOM_CMAP)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis(sweep_df, optimal_thresholds=None, title="Threshold Analysis", figsize=None):
    """
    Plot precision, recall, F1 across thresholds.
    
    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output from threshold_sweep().
    optimal_thresholds : dict, optional
        Dictionary of named thresholds to mark.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_MEDIUM
    
    plt.figure(figsize=figsize)
    
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color=MAIN_COLOR)
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color=SECONDARY_COLOR)
    plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", color=ACCENT_COLOR, linewidth=2)
    
    if optimal_thresholds:
        for name, t in optimal_thresholds.items():
            plt.axvline(t, linestyle="--", label=f"{name} (t={t:.2f})")
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cost_curve(thresholds, costs, optimal_threshold=None, title="Expected Cost vs Threshold", figsize=None):
    """
    Plot expected cost across thresholds.
    
    Parameters
    ----------
    thresholds : array-like
        Threshold values.
    costs : array-like
        Expected costs at each threshold.
    optimal_threshold : float, optional
        Optimal threshold to mark.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_SMALL
    
    plt.figure(figsize=figsize)
    
    plt.plot(thresholds, costs, color=MAIN_COLOR, linewidth=2)
    plt.axvline(0.5, linestyle="--", color="gray", label="Default (t=0.5)")
    
    if optimal_threshold is not None:
        plt.axvline(optimal_threshold, linestyle="--", color=ACCENT_COLOR, label=f"Optimal (t={optimal_threshold:.2f})")
    
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_category_effects(predictions_df, feature, target_col="proba", title=None, figsize=None):
    """
    Plot mean predicted probability by category.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and feature values.
    feature : str
        Categorical feature name.
    target_col : str
        Column with predicted probabilities.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_SMALL
    
    grouped = predictions_df.groupby(feature)[target_col].mean().sort_values(ascending=False)
    
    plt.figure(figsize=figsize)
    grouped.plot(kind="barh", color=MAIN_COLOR)
    plt.gca().invert_yaxis()
    
    title = title or f"Mean P(y=1) by {feature}"
    plt.title(title)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


# =============================================================================
# CLUSTERING VISUALIZATIONS
# =============================================================================
def plot_elbow_silhouette(k_range, inertias, silhouettes, best_k=None, figsize=None):
    """
    Plot elbow method and silhouette scores for cluster selection.
    
    Parameters
    ----------
    k_range : array-like
        Range of k values tested.
    inertias : array-like
        Inertia (WCSS) for each k.
    silhouettes : array-like
        Silhouette score for each k.
    best_k : int, optional
        Best k to highlight.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_WIDE
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Elbow plot
    axes[0].plot(list(k_range), inertias, "o-", color=MAIN_COLOR, markersize=8)
    if best_k is not None:
        best_idx = list(k_range).index(best_k)
        axes[0].axvline(best_k, linestyle="--", color=ACCENT_COLOR, label=f"Best k={best_k}")
        axes[0].legend()
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow Method")
    axes[0].set_xticks(list(k_range))
    
    # Silhouette plot
    axes[1].plot(list(k_range), silhouettes, "o-", color=MAIN_COLOR, markersize=8)
    if best_k is not None:
        axes[1].axvline(best_k, linestyle="--", color=ACCENT_COLOR, label=f"Best k={best_k}")
        axes[1].legend()
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].set_xticks(list(k_range))
    
    plt.tight_layout()
    plt.show()


def plot_cluster_scatter_pca(X, labels, title="Clusters in PCA Space", figsize=None):
    """
    Plot cluster assignments in PCA 2D space.
    
    Parameters
    ----------
    X : array-like
        Feature matrix (preferably scaled).
    labels : array-like
        Cluster assignments.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_MEDIUM
    
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=20, alpha=0.6, label=f"Cluster {label}", color=colors[i])
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(title)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_cluster_profiles(profile_df, title="Cluster Profiles (Mean Values)", figsize=None):
    """
    Plot cluster profiles as grouped bar chart.
    
    Parameters
    ----------
    profile_df : pd.DataFrame
        Mean feature values per cluster (clusters as rows, features as columns).
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or (12, 6)
    
    profile_df.T.plot(kind="bar", figsize=figsize, colormap="Set2")
    plt.xlabel("Feature")
    plt.ylabel("Mean Value")
    plt.title(title)
    plt.legend(title="Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_spending_mix(mix_df, cluster_names=None, figsize=None):
    """
    Plot spending mix (share of wallet) per cluster.
    
    Parameters
    ----------
    mix_df : pd.DataFrame
        Share of spending per category per cluster.
    cluster_names : dict, optional
        Mapping from cluster ID to name.
    figsize : tuple, optional
        Figure size.
    """
    n_clusters = len(mix_df)
    figsize = figsize or (6 * n_clusters, 4)
    
    fig, axes = plt.subplots(1, n_clusters, figsize=figsize)
    if n_clusters == 1:
        axes = [axes]
    
    for i, (idx, row) in enumerate(mix_df.iterrows()):
        name = cluster_names.get(idx, f"Cluster {idx}") if cluster_names else f"Cluster {idx}"
        row.plot(kind="bar", ax=axes[i], color=MAIN_COLOR)
        axes[i].set_title(f"Spending Mix — {name}")
        axes[i].set_ylabel("Share of Spend")
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_dendrogram(linkage_matrix, title="Hierarchical Clustering Dendrogram", truncate_level=5, figsize=None):
    """
    Plot dendrogram for hierarchical clustering.
    
    Parameters
    ----------
    linkage_matrix : array-like
        Linkage matrix from scipy.cluster.hierarchy.linkage().
    title : str
        Plot title.
    truncate_level : int
        Level at which to truncate dendrogram.
    figsize : tuple, optional
        Figure size.
    """
    from scipy.cluster.hierarchy import dendrogram
    
    figsize = figsize or FIGURE_SIZE_MEDIUM
    
    plt.figure(figsize=figsize)
    dendrogram(linkage_matrix, truncate_mode="level", p=truncate_level)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


# =============================================================================
# DEEP LEARNING VISUALIZATIONS
# =============================================================================
def plot_training_history(history, metrics=["loss", "accuracy"], title="Training History", figsize=None):
    """
    Plot training and validation metrics across epochs.
    
    Parameters
    ----------
    history : keras.callbacks.History or dict
        Training history object.
    metrics : list
        Metrics to plot.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    figsize = figsize or FIGURE_SIZE_WIDE
    
    # Handle both keras History object and dict
    if hasattr(history, "history"):
        history = history.history
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in history:
            axes[i].plot(history[metric], label=f"Train {metric}", color=MAIN_COLOR)
        val_metric = f"val_{metric}"
        if val_metric in history:
            axes[i].plot(history[val_metric], label=f"Val {metric}", color=SECONDARY_COLOR)
        
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f"{metric.capitalize()}")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
