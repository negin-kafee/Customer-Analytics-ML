# Notebooks

This directory contains Jupyter notebooks for the Customer Analytics ML pipeline. The notebooks should be run in the following order:

## 📚 Notebook Overview

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| **01_eda.ipynb** | Exploratory Data Analysis | Data insights, feature relationships |
| **02_regression.ipynb** | Spending Prediction Models | Best regression model (R² = 0.970) |
| **02b_regression_new_customers.ipynb** | New Customer Spending Prediction | Model for customers without purchase history |
| **03_classification.ipynb** | Campaign Response Prediction | Best classification model (ROC-AUC = 0.875) |
| **04_clustering.ipynb** | Customer Segmentation | Customer segments and insights |
| **05_deep_learning.ipynb** | Neural Network Models | Deep learning alternatives |

## 🚀 Getting Started

1. **Activate your environment:**
   ```bash
   source .venv/bin/activate  # or conda activate your-env
   ```

2. **Install dependencies:**
   ```bash
   cd ..  # Go to project root
   pip install -e .
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook  # or jupyter lab
   ```

4. **Run notebooks in order** (start with 01_eda.ipynb)

## 📊 Expected Results

After running all notebooks, you should have:
- **Trained models** saved in `../models/`
- **Analysis reports** saved in `../reports/`
- **Performance metrics** for all model types
- **Customer segments** with business insights

## 💡 Tips

- **Data file**: Ensure `../Data/marketing_campaign.csv` exists before starting
- **Memory**: Some notebooks are memory-intensive; close others when running
- **Reproducibility**: All notebooks use `RANDOM_STATE = 42` for consistent results
- **Errors**: If you encounter import errors, ensure the package is installed with `pip install -e .` from the project root

## 🔧 Troubleshooting

**Import errors?**
```bash
# From project root
pip install -e .
```

**Missing data?**
```bash
# Check data file exists
ls -la Data/marketing_campaign.csv
```

**Kernel issues?**
```bash
# Install kernel in your environment
python -m ipykernel install --user --name=customer-analytics
```