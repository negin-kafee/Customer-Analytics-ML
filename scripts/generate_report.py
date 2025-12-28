#!/usr/bin/env python3
"""
Generate a comprehensive project report from model outputs.

This script reads saved models and generates a summary report
with key metrics and insights.

Usage:
    python scripts/generate_report.py [--output OUTPUT_FILE]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import joblib


def load_model_metadata():
    """Load metadata from saved models."""
    models_dir = Path("models")
    metadata = {}
    
    # Load regression metadata
    reg_meta_path = models_dir / "regression_metadata.joblib"
    if reg_meta_path.exists():
        metadata["regression"] = joblib.load(reg_meta_path)
    
    # Load classification metadata  
    clf_meta_path = models_dir / "classification_metadata.joblib"
    if clf_meta_path.exists():
        metadata["classification"] = joblib.load(clf_meta_path)
        
    # Load clustering metadata
    cluster_meta_path = models_dir / "clustering_metadata.joblib"
    if cluster_meta_path.exists():
        metadata["clustering"] = joblib.load(cluster_meta_path)
    
    return metadata


def generate_markdown_report(metadata: dict) -> str:
    """Generate markdown report from metadata."""
    
    report = f"""# Customer Analytics ML - Model Performance Report

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

This report summarizes the performance of machine learning models trained for customer analytics tasks.

"""
    
    # Regression section
    if "regression" in metadata:
        reg_data = metadata["regression"]
        best_model = max(reg_data.get("cv_results", {}), key=lambda x: reg_data["cv_results"][x]["mean_score"])
        best_score = reg_data["cv_results"][best_model]["mean_score"]
        
        report += f"""## 💰 Spending Prediction (Regression)

**Best Model**: {best_model}  
**R² Score**: {best_score:.3f}  
**Cross-Validation Std**: {reg_data["cv_results"][best_model]["std_score"]:.3f}

### Top 3 Models:
"""
        # Sort by score and take top 3
        sorted_models = sorted(reg_data.get("cv_results", {}).items(), 
                             key=lambda x: x[1]["mean_score"], reverse=True)[:3]
        
        for i, (model, scores) in enumerate(sorted_models, 1):
            report += f"{i}. **{model}**: R² = {scores['mean_score']:.3f} (±{scores['std_score']:.3f})\n"
        
        report += "\n"
    
    # Classification section
    if "classification" in metadata:
        clf_data = metadata["classification"]
        best_model = max(clf_data.get("cv_results", {}), key=lambda x: clf_data["cv_results"][x]["mean_score"])
        best_score = clf_data["cv_results"][best_model]["mean_score"]
        
        report += f"""## 📧 Campaign Response Prediction (Classification)

**Best Model**: {best_model}  
**ROC-AUC Score**: {best_score:.3f}  
**Cross-Validation Std**: {clf_data["cv_results"][best_model]["std_score"]:.3f}

### Top 3 Models:
"""
        sorted_models = sorted(clf_data.get("cv_results", {}).items(), 
                             key=lambda x: x[1]["mean_score"], reverse=True)[:3]
        
        for i, (model, scores) in enumerate(sorted_models, 1):
            report += f"{i}. **{model}**: ROC-AUC = {scores['mean_score']:.3f} (±{scores['std_score']:.3f})\n"
        
        report += "\n"
    
    # Clustering section
    if "clustering" in metadata:
        cluster_data = metadata["clustering"]
        silhouette_score = cluster_data.get("silhouette_score", "N/A")
        n_clusters = cluster_data.get("n_clusters", "N/A")
        
        report += f"""## 👥 Customer Segmentation (Clustering)

**Algorithm**: K-Means  
**Number of Clusters**: {n_clusters}  
**Silhouette Score**: {silhouette_score}

### Cluster Insights:
*See notebooks/04_clustering.ipynb for detailed cluster analysis*
"""

    report += f"""
## Technical Details

### Environment
- **Python Version**: 3.10+
- **Key Libraries**: scikit-learn, XGBoost, TensorFlow
- **Cross-Validation**: 5-fold stratified

### Data
- **Dataset Size**: 2,240 customers
- **Features**: 29 original → 20+ engineered
- **Missing Values**: Handled via median imputation
- **Outliers**: Capped using IQR method

### Model Training
- **Hyperparameter Tuning**: Grid Search with CV
- **Class Imbalance**: Handled with SMOTE and class weights
- **Feature Selection**: Correlation and VIF analysis
- **Validation**: Stratified K-Fold cross-validation

---

*For detailed analysis, see the individual notebook reports in the `notebooks/` directory.*
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate model performance report")
    parser.add_argument(
        "--output", 
        type=Path,
        default=Path("MODEL_REPORT.md"),
        help="Output file path for the report"
    )
    args = parser.parse_args()
    
    print("🔍 Loading model metadata...")
    metadata = load_model_metadata()
    
    if not metadata:
        print("⚠️  No model metadata found. Run the notebooks first to generate models.")
        return
    
    print("📝 Generating report...")
    report = generate_markdown_report(metadata)
    
    print(f"💾 Saving report to {args.output}")
    args.output.write_text(report, encoding='utf-8')
    
    print("✅ Report generated successfully!")
    print(f"📄 View your report: {args.output}")


if __name__ == "__main__":
    main()