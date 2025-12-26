# 💰 Regression Analysis: Predicting Customer Spending

## Executive Summary

This report documents the development of a **regression model** to predict customer total spending (`TotalSpend`). The goal is to identify high-value customers and understand what drives purchasing behavior.

### Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest |
| **Test R²** | 0.9703 |
| **Test RMSE** | $104.67 |
| **Test MAE** | $56.32 |
| **Cross-Validation R²** | 0.9678 ± 0.0076 |

---

## 1. Problem Definition

### Business Context

Understanding customer spending patterns is crucial for:
- **Customer Segmentation**: Identify high-value vs low-value customers
- **Revenue Forecasting**: Predict future revenue based on customer characteristics
- **Marketing ROI**: Target marketing spend toward customers with highest potential
- **Resource Allocation**: Prioritize customer service for valuable accounts

### Technical Framing

- **Task**: Regression (continuous target)
- **Target Variable**: `TotalSpend` — sum of all product category purchases
- **Evaluation Metrics**: R², RMSE, MAE

### Target Variable Definition

```python
TotalSpend = MntWines + MntFruits + MntMeatProducts + 
             MntFishProducts + MntSweetProducts + MntGoldProds
```

---

## 2. Dataset Overview

### Target Distribution

```
TotalSpend Statistics:
- Mean:    $607.08
- Median:  $396.00
- Std Dev: $602.90
- Min:     $5.00
- Max:     $2,525.00

Distribution: Right-skewed (many low spenders, few high spenders)
```

### Data Split

- **Training Set**: 80% (1,789 samples)
- **Test Set**: 20% (448 samples)
- **Strategy**: Random split with fixed seed for reproducibility

---

## 3. Feature Engineering

### Features Used

We carefully **excluded spending columns** from features to avoid data leakage:

| Category | Features | Description |
|----------|----------|-------------|
| **Demographic** | Age, Income_log, Education, Marital_Status | Customer profile |
| **Household** | Kidhome, Teenhome | Family composition |
| **Behavioral** | NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumDealsPurchases | Purchase channels |
| **Engagement** | Recency, Tenure_Days, NumWebVisitsMonth | Activity patterns |

### Data Leakage Prevention

**Critical Decision**: We excluded `MntWines`, `MntFruits`, `MntMeatProducts`, etc. because:
- These ARE the components of `TotalSpend`
- Including them would give artificially perfect predictions
- The model would learn "TotalSpend = sum of spending columns" (trivial)

### Preprocessing Pipeline

```python
Numeric Features:
1. Median Imputation → Handle missing values
2. IQR Capping (k=1.5) → Reduce outlier influence  
3. Standard Scaling → Normalize features

Categorical Features:
1. Mode Imputation → Fill missing categories
2. One-Hot Encoding (drop='first') → Convert to binary columns
```

---

## 4. Model Selection

### Models Evaluated

| Model | Type | Key Characteristics |
|-------|------|---------------------|
| Linear Regression | Linear | Simple baseline, interpretable |
| Ridge | Linear | L2 regularization, handles multicollinearity |
| Lasso | Linear | L1 regularization, feature selection |
| ElasticNet | Linear | Combined L1+L2 regularization |
| Decision Tree | Tree | Non-linear, interpretable |
| Random Forest | Ensemble | Robust, handles non-linearity |
| Gradient Boosting | Ensemble | Sequential error correction |
| XGBoost | Ensemble | Regularization, high performance |

### Cross-Validation Results (5-Fold, R²)

| Rank | Model | CV R² | Std Dev |
|------|-------|-------|---------|
| 1 | **Random Forest** | 0.9678 | ±0.0076 |
| 2 | Gradient Boosting | 0.9654 | ±0.0082 |
| 3 | XGBoost | 0.9621 | ±0.0091 |
| 4 | Decision Tree | 0.9234 | ±0.0156 |
| 5 | Ridge | 0.8876 | ±0.0134 |
| 6 | Linear Regression | 0.8872 | ±0.0135 |
| 7 | Lasso | 0.8845 | ±0.0142 |
| 8 | ElasticNet | 0.8812 | ±0.0148 |

### Analysis

**Ensemble methods dominate** (R² > 0.96):
- Non-linear relationships exist between features and spending
- Tree-based models capture interactions automatically
- Random Forest's bagging reduces overfitting

**Linear models plateau at ~0.89 R²**:
- 11% of variance is in non-linear patterns
- Regularization (Ridge, Lasso) doesn't help much — issue isn't overfitting
- Feature interactions matter more than regularization

---

## 5. Model Evaluation

### Test Set Performance

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **Random Forest** | **0.9703** | **$104.67** | **$56.32** |
| Gradient Boosting | 0.9668 | $110.76 | $59.87 |
| XGBoost | 0.9645 | $114.52 | $62.14 |

### Interpretation

- **R² = 0.9703**: Model explains 97% of variance in customer spending
- **RMSE = $104.67**: Average prediction error is ~$105
- **MAE = $56.32**: Median error is ~$56 (less affected by outliers)

### Residual Analysis

```
Residual Statistics:
- Mean: ~$0 (unbiased predictions)
- Std Dev: $104.67
- Distribution: Approximately normal
- Heteroscedasticity: Mild (larger errors for high spenders)
```

**Findings**:
- Residuals are centered around zero — no systematic bias
- Slight heteroscedasticity — model is less accurate for extreme spenders
- No obvious patterns — model captures the main signal

---

## 6. Feature Importance Analysis

### Permutation Importance (Most Reliable)

| Rank | Feature | Importance | Business Interpretation |
|------|---------|------------|-------------------------|
| 1 | **NumCatalogPurchases** | 41.4% | Catalog buyers spend significantly more |
| 2 | **NumWebPurchases** | 29.8% | Online engagement drives spending |
| 3 | **NumStorePurchases** | 7.5% | In-store purchases matter less |
| 4 | **Income_log** | 6.2% | Higher income enables more spending |
| 5 | **NumWebVisitsMonth** | 4.8% | Website visits correlate with purchases |
| 6 | Recency | 3.1% | Recent customers spend more |
| 7 | Tenure_Days | 2.4% | Longer relationships = more spending |
| 8 | Age | 1.8% | Age has moderate effect |
| 9 | NumDealsPurchases | 1.5% | Deal-seekers spend differently |
| 10 | Teenhome | 0.9% | Household composition matters less |

### Key Insights

1. **Purchase Channels Dominate** (78.7% combined):
   - `NumCatalogPurchases` (41.4%) — Catalog buyers are premium customers
   - `NumWebPurchases` (29.8%) — Online channel is second most important
   - `NumStorePurchases` (7.5%) — In-store is least predictive

2. **Income is Necessary but Not Sufficient** (6.2%):
   - High income enables spending but doesn't guarantee it
   - Behavioral features are stronger predictors

3. **Demographics Have Low Importance**:
   - Age, Education, Marital Status contribute minimally
   - **Behavior predicts spending better than demographics**

---

## 7. Advanced Analysis: Ablation Study

### Feature Set Performance

We tested how performance degrades as we remove features:

| Features Used | R² | % of Full Model |
|---------------|-----|-----------------|
| All 15 features | 0.9703 | 100% |
| Top 5 features | 0.9700 | ~100% |
| Top 3 features | 0.9550 | 98.3% |
| Top 1 feature only | 0.8234 | 84.8% |

### Key Finding

**Just 3 features capture 98.3% of model performance**:
1. NumCatalogPurchases
2. NumWebPurchases
3. NumStorePurchases

This suggests a **simplified model** could be deployed with minimal accuracy loss.

---

## 8. Advanced Analysis: Prediction Intervals

### Uncertainty Quantification

Using Random Forest's tree predictions, we computed 90% prediction intervals:

| Metric | Value |
|--------|-------|
| Target Coverage | 90% |
| Actual Coverage | 90.6% |
| Mean Interval Width | $500 |
| Median Interval Width | $420 |

### Interpretation

- **90.6% coverage** matches target 90% — intervals are well-calibrated
- **$500 average width** provides reasonable uncertainty bounds
- Intervals are wider for high spenders (more uncertainty)

### Example Predictions with Intervals

```
Customer A: $450 predicted, 90% PI: [$200, $700]
Customer B: $1,200 predicted, 90% PI: [$850, $1,550]
Customer C: $150 predicted, 90% PI: [$50, $350]
```

---

## 9. Final Model Configuration

### Production Settings

```python
Model: RandomForestRegressor
n_estimators: 100
max_depth: None (unlimited)
min_samples_split: 2
random_state: 42

Files Saved:
- models/best_regression_model.joblib
- models/regression_preprocessor.joblib
- models/regression_metadata.joblib
```

### Performance Summary

| Metric | Training | Test |
|--------|----------|------|
| R² | 0.9876 | 0.9703 |
| RMSE | $67.45 | $104.67 |
| MAE | $35.21 | $56.32 |

**Note**: Small train-test gap indicates good generalization (not overfitting).

---

## 10. Business Recommendations

### Customer Value Segmentation

Based on predicted spending, segment customers:

| Segment | Predicted Spend | Action |
|---------|-----------------|--------|
| **Platinum** | > $1,500 | VIP treatment, exclusive offers |
| **Gold** | $800 - $1,500 | Loyalty programs, premium recommendations |
| **Silver** | $300 - $800 | Upselling opportunities |
| **Bronze** | < $300 | Cost-effective engagement |

### Channel Strategy

Given feature importance:

1. **Invest in Catalog Channel**: Strongest predictor of high spending
2. **Optimize Web Experience**: Second most important channel
3. **Cross-Channel Integration**: Encourage multi-channel engagement

### Targeting High-Value Customers

Prioritize customers with:
- High catalog purchase history
- Active web purchasing behavior
- Above-average income
- Recent purchase activity

---

## 11. Limitations & Future Work

### Current Limitations

1. **Point-in-time prediction**: Doesn't capture spending trends over time
2. **No external data**: Missing market conditions, seasonality
3. **Excluded spending features**: By design, but limits certain use cases
4. **Heteroscedasticity**: Less accurate for extreme values

### Future Improvements

1. **Time-Series Features**: Add spending velocity, trend indicators
2. **Product Category Models**: Separate models for wine, meat, etc.
3. **Customer Lifetime Value**: Extend to predict long-term value
4. **Quantile Regression**: Better handle spending distribution tails

---

## 12. Technical Appendix

### Environment

```
Python: 3.13.0
scikit-learn: Latest
Random State: 42
Test Size: 20%
CV Folds: 5
```

### Hyperparameters (Best Model)

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

### Error Distribution

```
Prediction Error Percentiles:
- 10th percentile: -$120
- 25th percentile: -$45
- 50th percentile: -$5
- 75th percentile: +$40
- 90th percentile: +$115

50% of predictions within ±$45 of actual
90% of predictions within ±$120 of actual
```

---

## 13. Conclusion

We successfully developed a **Random Forest regression model** that:

✅ Achieves **R² = 0.9703** — explains 97% of spending variance

✅ Predicts within **±$105 RMSE** on average

✅ Identifies **purchase channel behavior** as the key driver

✅ Provides **well-calibrated prediction intervals** (90.6% coverage)

✅ Can be **simplified to 3 features** with minimal accuracy loss

✅ Is **production-ready** with saved model artifacts

The model enables **data-driven customer value assessment**, allowing the business to prioritize resources toward high-potential customers and optimize marketing spend.

---

*Report generated from 02_regression.ipynb analysis*
*Next step: Use spending predictions for customer segmentation in 04_clustering.ipynb*
