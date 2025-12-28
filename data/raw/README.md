# Raw Data

This directory contains the original, unmodified datasets for the Customer Analytics ML project.

## Dataset: Customer Personality Analysis

**Source**: [Kaggle - Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)  
**License**: CC0 (Public Domain)  
**File**: `marketing_campaign.csv`

### Dataset Overview

- **Records**: 2,240 customers
- **Features**: 29 variables
- **Format**: Comma-separated values (CSV)
- **Size**: ~220KB

### Key Features

| Category | Features |
|----------|----------|
| **Demographics** | Age, Income, Education, Marital Status |
| **Family** | Children at home, Teenagers at home |
| **Spending** | Wine, Fruits, Meat, Fish, Sweets, Gold |
| **Purchasing** | Store, Web, Catalog purchases |
| **Campaigns** | Response to marketing campaigns |
| **Behavioral** | Days since last purchase, complaints |

### Usage

The data is automatically loaded by notebooks using the path configured in `src/config.py`:
```python
DATA_PATH = "data/raw/marketing_campaign.csv"
```

### Data Quality

- **Missing values**: Some records have missing income data
- **Outliers**: Present in spending and income variables  
- **Data types**: Mixed numeric and categorical variables

All data quality issues are handled in the preprocessing pipeline.