# Data Directory

This directory contains all data files for the Customer Analytics ML project, organized by processing stage.

## 📁 Directory Structure

```
data/
├── raw/                    # Original, immutable data
├── interim/               # Intermediate data (partially processed)
└── processed/            # Final, analysis-ready datasets
```

## 📋 Data Stages

### `/raw/`
- **Purpose**: Original, unmodified datasets
- **Files**: 
  - `marketing_campaign.csv` - Main customer dataset
  - `clustered_customers.csv` - Data with cluster assignments
- **Rule**: Never modify files in this directory

### `/interim/`
- **Purpose**: Partially processed data
- **Examples**: 
  - Data after initial cleaning
  - Feature-engineered datasets before final preprocessing
  - Temporary files during processing steps

### `/processed/`
- **Purpose**: Final, analysis-ready datasets
- **Examples**:
  - Train/test splits
  - Scaled and preprocessed features
  - Model-ready datasets

## 🔄 Data Flow

```
Raw Data → EDA/Cleaning → Interim Data → Final Processing → Processed Data → Models
```

## 📊 Dataset Information

**Primary Dataset**: `marketing_campaign.csv`
- **Size**: 2,240 customers
- **Features**: 29 variables (demographic, behavioral, campaign response)
- **Target Variables**: 
  - Spending amount (regression)
  - Campaign acceptance (classification)
  - Customer segments (clustering)

## 🔒 Data Guidelines

1. **Never modify raw data** - Keep original files unchanged
2. **Document transformations** - Comment your preprocessing steps
3. **Version control** - Track data changes when appropriate
4. **Backup important results** - Save processed datasets for reuse