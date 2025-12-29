# 📊 Exploratory Data Analysis: Marketing Campaign Dataset

---

**Notebook**: `01_eda.ipynb`  
**Version**: 1.0  
**Last Updated**: December 2024

---

## Executive Summary

This report documents the **Exploratory Data Analysis (EDA)** performed on the marketing campaign dataset. The goal is to understand customer characteristics, identify data quality issues, and discover patterns that inform downstream modeling.

### Key Findings

| Metric | Value |
|--------|-------|
| **Total Customers** | 2,240 |
| **Features** | 29 columns |
| **Missing Values** | 24 in Income (1.1%) |
| **Outliers Detected** | Age (3), Income (8), Spending columns |
| **Target Imbalance** | Response: 15% positive (5.7:1 ratio) |

---

## 1. Dataset Overview

### Data Source

The dataset contains customer information from a marketing campaign, including:
- **Demographics**: Age, Education, Marital Status, Income
- **Household**: Number of children (Kidhome, Teenhome)
- **Purchase Behavior**: Spending by product category, purchase channels
- **Engagement**: Recency, website visits, campaign responses

### Initial Shape

```
Rows: 2,240 customers
Columns: 29 features
```

### Column Categories

| Category | Columns | Count |
|----------|---------|-------|
| **ID** | ID | 1 |
| **Demographics** | Year_Birth, Education, Marital_Status, Income | 4 |
| **Household** | Kidhome, Teenhome | 2 |
| **Spending** | MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds | 6 |
| **Purchases** | NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases | 4 |
| **Engagement** | NumWebVisitsMonth, Recency | 2 |
| **Campaign** | AcceptedCmp1-5, Response | 6 |
| **Dates** | Dt_Customer | 1 |
| **Other** | Complain, Z_CostContact, Z_Revenue | 3 |

---

## 2. Data Quality Assessment

### Missing Values

```
Column        Missing    Percentage
Income           24         1.07%
All others        0         0.00%
```

**Action**: Impute Income with median (robust to outliers)

**⚠️ Data Leakage Note**: In the EDA notebook, imputation uses the full dataset median for visualization. For production models, imputation is performed **after train-test split** using only training data statistics.

### Constant Columns

```
Z_CostContact: 1 unique value (3)
Z_Revenue:     1 unique value (11)
```

**Action**: Drop these columns — no predictive value with zero variance.

### Duplicate Records

```
Duplicate rows: 0
```

Dataset is clean with no duplicate entries.

### Data Validation

- ✓ No negative values in spending columns
- ✓ No negative values in purchase count columns
- ✓ Age outliers identified (3 records with unrealistic ages)

### Data Types

| Type | Columns |
|------|---------|
| Numeric (int64) | 25 |
| Numeric (float64) | 1 (Income) |
| Object (string) | 3 (Education, Marital_Status, Dt_Customer) |

---

## 3. Univariate Analysis

### Numeric Features

#### Age Distribution

```
Derived from: 2024 - Year_Birth

Statistics:
- Mean: 52.2 years
- Median: 51 years
- Std Dev: 11.9 years
- Min: 27 years
- Max: 131 years (outlier!)

Outliers: 3 customers with Age > 100 (likely data entry errors)
Action: Filter Age <= 100
```

#### Income Distribution

```
Statistics:
- Mean: $52,247
- Median: $51,382
- Std Dev: $25,173
- Min: $1,730
- Max: $666,666 (extreme outlier!)

Distribution: Right-skewed
Outliers: 8 customers with Income > $150,000
Action: Log-transform for modeling, cap extreme values
```

#### Spending Distributions

| Product | Mean | Median | Max | Skewness |
|---------|------|--------|-----|----------|
| MntWines | $303 | $174 | $1,493 | Right |
| MntMeatProducts | $167 | $68 | $1,725 | Right |
| MntGoldProds | $44 | $24 | $362 | Right |
| MntFishProducts | $37 | $12 | $259 | Right |
| MntSweetProducts | $27 | $8 | $263 | Right |
| MntFruits | $26 | $8 | $199 | Right |

**Key Insight**: All spending columns are right-skewed — most customers spend little, few spend a lot.

#### Total Spend (Derived)

```
TotalSpend = Sum of all Mnt* columns

Statistics:
- Mean: $607
- Median: $396
- Std Dev: $603
- Min: $5
- Max: $2,525
```

### Categorical Features

#### Education

```
Distribution:
- Graduation:    50.1%
- PhD:           21.6%
- Master:        17.0%
- 2n Cycle:       9.0%
- Basic:          2.3%

Insight: Highly educated customer base (>70% with degree)
```

#### Marital Status

```
Distribution:
- Married:       38.6%
- Together:      26.0%
- Single:        21.4%
- Divorced:       9.8%
- Widow:          3.4%
- Other:          0.8%

Action: Consolidate rare categories (Alone, Absurd, YOLO → Other)
```

---

## 4. Bivariate Analysis

### Correlation Analysis

**Method**: Spearman correlation is used throughout as it's more robust to non-normal distributions and outliers typical in this dataset.

#### Top Positive Correlations

| Feature Pair | Correlation |
|--------------|-------------|
| NumCatalogPurchases ↔ MntMeatProducts | 0.72 |
| NumCatalogPurchases ↔ MntWines | 0.64 |
| Income ↔ MntWines | 0.58 |
| Income ↔ MntMeatProducts | 0.58 |
| NumStorePurchases ↔ MntWines | 0.55 |

#### Top Negative Correlations

| Feature Pair | Correlation |
|--------------|-------------|
| Kidhome ↔ MntWines | -0.50 |
| Kidhome ↔ MntMeatProducts | -0.41 |
| Kidhome ↔ Income | -0.32 |
| NumWebVisitsMonth ↔ NumStorePurchases | -0.28 |

### Key Relationships

1. **Catalog Purchases → High Spending**: Customers who buy via catalog spend significantly more on meat and wine

2. **Income → Premium Products**: Higher income correlates with wine and meat purchases

3. **Kids → Lower Spending**: Households with children spend less on premium categories

4. **Channel Substitution**: Web visits negatively correlate with store purchases (customers prefer one channel)

### Spending by Demographics

#### By Education

| Education | Avg TotalSpend |
|-----------|----------------|
| PhD | $720 |
| Master | $651 |
| Graduation | $598 |
| 2n Cycle | $478 |
| Basic | $245 |

**Insight**: Clear positive relationship between education and spending

#### By Marital Status

| Status | Avg TotalSpend |
|--------|----------------|
| Widow | $712 |
| Divorced | $665 |
| Single | $627 |
| Together | $571 |
| Married | $558 |

**Insight**: Single-income households spend more (possibly disposable income effect)

#### By Children

| Has Kids | Avg TotalSpend |
|----------|----------------|
| No | $829 |
| Yes | $386 |

**Insight**: Households without children spend 2.1× more

---

## 5. Target Variable Analysis

### Response (Campaign Response)

```
Distribution:
- No Response (0): 1,903 (85.1%)
- Response (1):      334 (14.9%)

Imbalance Ratio: 5.7:1
```

### Response Rate by Segment

| Segment | Response Rate |
|---------|--------------|
| High Income (>$70k) | 22.3% |
| Low Income (<$30k) | 8.1% |
| PhD Education | 18.4% |
| Basic Education | 6.2% |
| No Children | 21.8% |
| Has Children | 9.7% |
| High Spenders (>$800) | 26.4% |
| Low Spenders (<$200) | 7.2% |

**Key Insight**: Response rate varies significantly by segment — high-value customers respond more.

---

## 6. Statistical Tests

### Normality Tests (Shapiro-Wilk)

| Feature | W Statistic | p-value | Conclusion |
|---------|-------------|---------|------------|
| Income | 0.651 | <0.001 | NOT Normal |
| Age | 0.987 | <0.001 | NOT Normal |
| TotalSpend | 0.894 | <0.001 | NOT Normal |
| Recency | 0.953 | <0.001 | NOT Normal |

**Implication**: Use non-parametric methods (Spearman correlation) and consider log transformations.

### Chi-Square Tests (Categorical vs Response)

| Feature | χ² | p-value | Conclusion |
|---------|-----|---------|------------|
| Education | 47.2 | <0.001 | SIGNIFICANT |
| Marital_Status | 31.8 | <0.001 | SIGNIFICANT |
| HasChildren | 89.6 | <0.001 | SIGNIFICANT |

**Implication**: All categorical features have significant association with campaign response.

### ANOVA (Numeric Features by Response)

| Feature | F-statistic | p-value | Conclusion |
|---------|-------------|---------|------------|
| Income | 156.3 | <0.001 | SIGNIFICANT |
| TotalSpend | 201.4 | <0.001 | SIGNIFICANT |
| Recency | 45.7 | <0.001 | SIGNIFICANT |
| Age | 12.3 | <0.001 | SIGNIFICANT |

**Implication**: Responders significantly differ from non-responders across all numeric features.

### Multicollinearity (VIF Analysis)

| Feature | VIF | Status |
|---------|-----|--------|
| Income | 1.82 | OK |
| Age | 1.15 | OK |
| Recency | 1.03 | OK |
| NumWebPurchases | 2.34 | OK |
| NumCatalogPurchases | 2.89 | OK |
| NumStorePurchases | 2.12 | OK |

**Conclusion**: No severe multicollinearity (all VIF < 5). Features are safe to use together.

---

## 7. Feature Engineering Opportunities

# Feature Engineering (EDA Phase)

During the Exploratory Data Analysis (EDA) phase, we engineered a set of derived features to transform raw transactional and demographic variables into meaningful behavioral signals. The goal was to reduce dimensionality, normalize information across customers, and create interpretable features aligned with spending behavior, household context, engagement, and customer lifecycle.

These features were created **for exploratory understanding and hypothesis generation**. In the modeling phase, all transformations are re-implemented inside proper preprocessing pipelines to avoid data leakage.

---

### 1. Spending Features

Raw spending variables were aggregated and normalized to capture both total value and product preferences.

**Base variables**
- `MntWines`, `MntFruits`, `MntMeatProducts`
- `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`

**Engineered features**

| Feature | Formula | Purpose |
|---------|---------|---------|
| **TotalSpend** | Sum of all product spending categories | Represents overall monetary value of the customer |
| **TotalSpend_log** | Log-transformed total spending (`log(1 + TotalSpend)`) | Reduces skewness and stabilizes variance for regression modeling |
| **SpendingRatio** | Ratio of total spending to income | Indicates consumption intensity and spending aggressiveness |

---

### 2. Family and Normalization Features

Household-aware normalization was applied to account for differences in family size.

**Engineered features**

| Feature | Formula | Purpose |
|---------|---------|---------|
| **FamilySize** | Computed as `1 + Kidhome + Teenhome` | Total household members for normalization |
| **IncomePerCapita** | Income normalized by family size | More informative than raw income when comparing households |
| **HasChildren** | `(Kidhome + Teenhome) > 0` | Binary indicator of presence of children |
| **TotalChildren** | `Kidhome + Teenhome` | Total number of dependents |

---

### 3. Engagement and Purchase Behavior Features

Customer interaction behavior was summarized across channels to capture engagement patterns.

**Base variables**
- `NumWebPurchases`, `NumStorePurchases`, `NumCatalogPurchases`, `NumDealsPurchases`
- `NumWebVisitsMonth`, `Recency`

**Engineered features**

| Feature | Formula | Purpose |
|---------|---------|---------|
| **TotalPurchases** | Total number of purchases across all channels | Overall purchase frequency |
| **AvgSpendPerPurchase** | `TotalSpend / TotalPurchases` | Indicates average transaction value |
| **WebPurchaseRatio** | `NumWebPurchases / TotalPurchases` | Digital vs offline preference |

---

### 4. Customer Lifecycle and Temporal Features

These features capture temporal and demographic aspects of the customer journey.

**Engineered features**

| Feature | Formula | Purpose |
|---------|---------|---------|
| **Age** | `2024 - Year_Birth` | More interpretable than birth year |
| **Tenure_Days** | Days since customer enrollment | Customer relationship length |
| **Tenure_Months** | `Tenure_Days // 30` | Tenure in more interpretable units |

---

### 5. Campaign Response and Segmentation Features

Campaign engagement history was aggregated to understand customer responsiveness patterns.

**Base variables**
- `AcceptedCmp1`, `AcceptedCmp2`, `AcceptedCmp3`, `AcceptedCmp4`, `AcceptedCmp5`

**Engineered features**

| Feature | Formula | Purpose |
|---------|---------|---------|
| **TotalAccepted** | Sum of all campaign acceptances | Overall campaign responsiveness |
| **IsPreviousResponder** | `TotalAccepted > 0` | Binary indicator of any past response |
| **HasHistory** | `Tenure_Days > median` | Long-term vs newer customer classification |

---

### 6. Education and Demographic Encoding

Categorical variables were encoded to preserve ordinality and business meaning.

**Engineered features**

| Feature | Mapping | Purpose |
|---------|---------|---------|
| **Education_Num** | Basic=1, 2n Cycle=2, Graduation=3, Master=4, PhD=5 | Ordinal encoding preserving education hierarchy |
| **Marital_Status_Clean** | Rare categories → "Other" | Consolidation of low-frequency categories |
| **IsPartner** | Married/Together = 1, Others = 0 | Binary partnership indicator |

---

### 7. Outlier Treatment (EDA Only)

Outliers were identified using IQR-based methods and visual inspection.

- Extreme values were **capped, not removed**, to preserve high-value customers
- Capping thresholds computed during EDA are **not reused directly** in modeling
- All outlier handling is re-applied inside training pipelines using training data only

---

### 8. Missing Data Handling

- Only the `Income` variable contained missing values (~1%)
- Missingness was assessed and assumed **MCAR (Missing Completely At Random)**
- Median imputation was used during EDA for exploration
- Final imputation is performed inside preprocessing pipelines after train–test split

---

### Notes on Data Leakage Prevention

EDA was conducted on the full dataset to understand distributions and relationships. All feature transformations identified here are re-implemented during modeling using training-only statistics to prevent information leakage.

### Derived Features Created

| Feature | Formula | Purpose |
|---------|---------|---------|
| **Age** | 2024 - Year_Birth | More interpretable than birth year |
| **Tenure_Days** | Today - Dt_Customer | Customer relationship length |
| **TotalSpend** | Sum of Mnt* columns | Overall customer value |
| **TotalPurchases** | Sum of Num*Purchases | Purchase frequency |
| **Income_log** | log(Income) | Normalize skewed distribution |
| **HasChildren** | Kidhome + Teenhome > 0 | Binary flag |
| **TotalChildren** | Kidhome + Teenhome | Total dependents |
| **TotalAcceptedCmp** | Sum of AcceptedCmp* | Campaign engagement |

### Recommended Transformations

1. **Log Transform**: Income (highly skewed)
2. **Binning**: Age into generations (Gen Z, Millennial, Gen X, Boomer)
3. **Consolidation**: Rare marital status categories
4. **Ratios**: Spend per purchase, catalog vs web ratio

---

## 7. Outlier Analysis

### Detection Method: IQR Rule

```
Outlier if: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
```

### Outliers by Feature

| Feature | # Outliers | % of Data | Action |
|---------|------------|-----------|--------|
| Age | 3 | 0.13% | Remove (data errors) |
| Income | 8 | 0.36% | Cap at 99th percentile |
| MntWines | 12 | 0.54% | Cap (valid high spenders) |
| MntMeatProducts | 18 | 0.80% | Cap (valid high spenders) |
| MntGoldProds | 15 | 0.67% | Cap |

### Treatment Strategy

- **Age outliers**: Remove (impossible values like 131 years)
- **Income/Spending outliers**: Cap using IQR method (valid but extreme)
- **Purpose**: Reduce influence on model without losing information

---

## 8. Data Quality Summary

### Issues Identified

| Issue | Severity | Count | Resolution |
|-------|----------|-------|------------|
| Missing Income | Low | 24 | Median imputation |
| Age outliers | Medium | 3 | Remove rows |
| Income outliers | Low | 8 | IQR capping |
| Spending outliers | Low | ~50 | IQR capping |
| Rare categories | Low | 4 | Consolidate |

### Final Clean Dataset

```
After cleaning:
- Rows: 2,237 (removed 3 age outliers)
- Columns: 29 + 8 engineered = 37
- Missing: 0 (after imputation)
```

---

## 9. Key Insights for Modeling

### For Regression (Predicting Spending)

1. **Use purchase channel features** — strongest predictors
2. **Log-transform income** — handles skewness
3. **Include household composition** — significant impact on spending
4. **Exclude spending columns as features** — prevent leakage

### For Classification (Predicting Response)

1. **Class imbalance requires handling** — use stratified sampling, class weights
2. **High-value segments respond more** — feature engineering around value
3. **ROC-AUC and PR-AUC are key metrics** — not accuracy
4. **Threshold tuning needed** — default 0.5 is suboptimal

### For Clustering (Customer Segmentation)

1. **Scale features** — different units ($ vs counts)
2. **Consider RFM framework** — Recency, Frequency, Monetary
3. **Multiple cluster solutions** — test k=3,4,5,6
4. **Behavioral features may outperform demographics**

---

## 10. Visualization Summary

### Key Plots Generated

1. **Distribution Plots**: All numeric features — identified skewness
2. **Correlation Heatmap**: Feature relationships — identified multicollinearity
3. **Box Plots**: Spending by demographics — identified segment differences
4. **Scatter Plots**: Income vs Spending — confirmed positive relationship
5. **Bar Charts**: Category distributions — education and marital status
6. **Target Analysis**: Response rates by segment — identified high-value segments

---

## 11. Recommendations

### Data Collection

1. **Validate birth year entries** — impossible ages suggest data entry issues
2. **Cap income at realistic max** — $666K is likely an error
3. **Add timestamp data** — enable time-series analysis

### Feature Engineering

1. **Create RFM scores** — standard customer value framework
2. **Add channel preference ratio** — web vs catalog vs store
3. **Compute customer lifetime value** — combine with tenure

### Analysis Extensions

1. **Cohort analysis** — behavior by customer vintage
2. **Product affinity** — which products are purchased together
3. **Seasonal patterns** — if time data available

---

## 12. Technical Appendix

### Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

### Dataset Schema

```
ID                    int64     # Customer identifier
Year_Birth            int64     # Birth year
Education            object     # Education level
Marital_Status       object     # Relationship status
Income              float64     # Annual household income
Kidhome               int64     # Children at home
Teenhome              int64     # Teenagers at home
Dt_Customer          object     # Enrollment date
Recency               int64     # Days since last purchase
MntWines              int64     # Wine spending
MntFruits             int64     # Fruit spending
MntMeatProducts       int64     # Meat spending
MntFishProducts       int64     # Fish spending
MntSweetProducts      int64     # Sweet spending
MntGoldProds          int64     # Gold products spending
NumDealsPurchases     int64     # Purchases with discount
NumWebPurchases       int64     # Web purchases
NumCatalogPurchases   int64     # Catalog purchases
NumStorePurchases     int64     # Store purchases
NumWebVisitsMonth     int64     # Monthly web visits
AcceptedCmp1-5        int64     # Campaign acceptance flags
Response              int64     # Response to last campaign
Complain              int64     # Complaint flag
Z_CostContact         int64     # Contact cost (constant)
Z_Revenue             int64     # Revenue (constant)
```

---

## 13. Conclusion

The EDA revealed a **high-quality dataset** with:

✅ Minimal missing data (1.1% in one column)

✅ Clear patterns between features and targets

✅ Actionable segments based on demographics and behavior

✅ Feature engineering opportunities for improved modeling

⚠️ Class imbalance requiring special handling

⚠️ Outliers needing treatment before modeling

The dataset is **well-suited for**:
- **Regression**: Predicting customer spending
- **Classification**: Predicting campaign response
- **Clustering**: Customer segmentation

---

*Report generated from 01_eda.ipynb analysis*
*Next step: Proceed to 02_regression.ipynb for spending prediction*
