# Customer Analytics ML - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-28

### Added
- Complete ML pipeline for customer analytics
- EDA notebook with comprehensive data analysis
- Regression models for spending prediction (R² = 0.970)
- Classification models for campaign response (ROC-AUC = 0.875)
- Customer segmentation using K-Means clustering
- Deep learning models using TensorFlow/Keras
- Custom preprocessing transformers
- Comprehensive evaluation metrics and logging
- Professional project structure with proper packaging
- Automated testing and linting setup
- Makefile for common development tasks

### Project Structure
- Moved notebooks to dedicated `notebooks/` directory
- Organized source code in `src/` package
- Added comprehensive documentation
- Implemented proper Python packaging with `pyproject.toml`
- Added development tools configuration

### Models Included
- Random Forest (best for regression and classification)
- XGBoost with hyperparameter optimization
- Support Vector Machines
- Gradient Boosting
- Multi-layer Perceptron (TensorFlow)
- K-Means clustering with optimization

### Features
- 15+ machine learning algorithms compared
- Custom sklearn transformers for reproducible preprocessing
- Hyperparameter optimization with cross-validation
- Imbalanced classification handling
- Comprehensive model evaluation and comparison
- Production-ready model persistence