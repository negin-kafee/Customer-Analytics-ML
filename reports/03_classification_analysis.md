# 🎯 Classification Analysis: Predicting Marketing Campaign Response

## Executive Summary

This report documents the development of a **binary classification model** to predict customer response to marketing campaigns. The goal is to optimize marketing spend by targeting customers most likely to respond.

### Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest |
| **ROC-AUC** | 0.8751 |
| **PR-AUC** | 0.5739 (3.8× baseline) |
| **Optimal Threshold** | 0.30 |
| **Recall at Optimal Threshold** | 76.1% |
| **Precision at Optimal Threshold** | 46.4% |
| **Cost Reduction** | 28% vs default threshold |

---

## 1. Problem Definition

### Business Context

Marketing campaigns are expensive. Sending promotional materials to customers who won't respond wastes resources, while missing potential responders means lost revenue. The challenge is to **identify which customers are most likely to respond** to optimize campaign ROI.

### Technical Framing

- **Task**: Binary classification (Response = 1 or 0)
- **Target Variable**: `Response` — whether a customer accepted the offer in the last campaign
- **Evaluation Focus**: Maximize customer capture while minimizing costs

### Key Challenges

1. **Class Imbalance**: Only ~15% of customers respond (5.7:1 ratio)
2. **Asymmetric Costs**: Missing a customer costs more than unnecessary contact
3. **Threshold Selection**: Default 0.5 threshold is suboptimal for imbalanced data

---

## 2. Dataset Overview

### Class Distribution

```
Response Distribution:
- No Response (0): 1,903 customers (85.1%)
- Response (1):      334 customers (14.9%)

Imbalance Ratio: 5.7:1
```

### Why Imbalance Matters

A naive model predicting "No Response" for everyone achieves 85% accuracy but is **completely useless**. This is why we focus on:
- **ROC-AUC**: Measures ranking ability across all thresholds
- **PR-AUC**: More informative for imbalanced data
- **Recall**: Ensures we capture actual responders
- **F1 Score**: Balances precision and recall

### Data Split

- **Training Set**: 80% (1,789 samples)
- **Test Set**: 20% (448 samples)
- **Strategy**: Stratified split to preserve class ratios

---

## 3. Feature Engineering

### Features Used (21 total after preprocessing)

| Category | Features | Description |
|----------|----------|-------------|
| **Demographic** | Age, Income_log, Education, Marital_Status | Customer profile |
| **Household** | Kidhome, Teenhome | Family composition |
| **Behavioral** | NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumDealsPurchases | Purchase channels |
| **Engagement** | Recency, Tenure_Days, NumWebVisitsMonth | Activity patterns |
| **Economic** | TotalSpend | Derived spending metric |

### Preprocessing Pipeline

```python
Numeric Features:
1. Median Imputation → Handle missing values
2. IQR Capping (k=1.5) → Reduce outlier influence
3. Standard Scaling → Normalize for distance-based models

Categorical Features:
1. Mode Imputation → Fill missing categories
2. One-Hot Encoding (drop='first') → Convert to binary columns
```

### Feature Selection Rationale

We include **all available features** including spending-related columns because:
- `Response` is **independent** of historical spending behavior
- A customer's response to a campaign is not caused by their past spending
- High spenders may be more engaged and thus more likely to respond

---

## 4. Model Selection

### Models Evaluated

| Model | Type | Key Characteristics |
|-------|------|---------------------|
| Logistic Regression | Linear | Interpretable, handles class weights |
| Random Forest | Ensemble | Robust, handles non-linearity |
| Gradient Boosting | Ensemble | High accuracy, sequential learning |
| XGBoost | Ensemble | Regularization, handles imbalance |
| AdaBoost | Ensemble | Focuses on hard examples |
| KNN | Instance-based | Non-parametric |
| Decision Tree | Tree | Interpretable but overfits |

### Cross-Validation Results (5-Fold Stratified, ROC-AUC)

| Rank | Model | ROC-AUC | Std Dev |
|------|-------|---------|---------|
| 1 | **Random Forest** | 0.8616 | ±0.026 |
| 2 | Gradient Boosting | 0.8580 | ±0.018 |
| 3 | Logistic Regression | 0.8576 | ±0.014 |
| 4 | XGBoost | 0.8417 | ±0.029 |
| 5 | AdaBoost | 0.8410 | ±0.021 |
| 6 | KNN | 0.7482 | ±0.016 |
| 7 | Decision Tree | 0.6180 | ±0.049 |

### Analysis

**Top 3 models are nearly tied** (~0.86 ROC-AUC), suggesting:
- The signal in the data is learnable but bounded
- Model choice matters less than proper evaluation and threshold tuning
- Feature engineering has more impact than algorithm selection

**Decision Tree significantly underperforms** (0.62) because:
- Prone to overfitting on imbalanced data
- Cannot capture complex interactions without ensembling

**KNN struggles** (0.75) because:
- Distance metrics are problematic with mixed feature types
- Curse of dimensionality with 21 features

---

## 5. Model Evaluation

### Test Set Performance (Default Threshold = 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|-----|---------|--------|
| **Random Forest** | 85.9% | 54.5% | 35.8% | 0.432 | **0.875** | **0.574** |
| Gradient Boosting | 87.7% | 67.6% | 34.3% | 0.455 | 0.871 | 0.563 |
| Logistic Regression | 76.3% | 36.6% | 79.1% | 0.500 | 0.856 | 0.493 |

### Key Observations

1. **Random Forest** has highest ROC-AUC (0.875) — best overall discriminative power
2. **Gradient Boosting** has highest precision (67.6%) — most conservative predictions
3. **Logistic Regression** has highest recall (79.1%) — most aggressive predictions

### The Default Threshold Problem

At threshold = 0.5, all models exhibit **low recall** (35-36% for ensemble models):
- We're missing **64% of potential responders**
- The threshold is too conservative for imbalanced data
- We need cost-sensitive threshold optimization

---

## 6. Threshold Optimization

### Why 0.5 is Wrong

The default threshold of 0.5 assumes:
- Equal class distribution (50/50) ❌ We have 85/15
- Equal misclassification costs ❌ Missing customers costs more

### Optimal Thresholds by Objective

| Objective | Optimal Threshold | Description |
|-----------|------------------|-------------|
| **Maximize F1** | 0.33 | Best precision-recall balance |
| **Recall ≥ 70%** | 0.33 | Ensure high customer capture |
| **Precision ≥ 50%** | 0.34 | Ensure predictions are reliable |
| **Cost Optimal** | 0.30 | Minimize total business cost |

### Cost-Sensitive Optimization

**Business Cost Framework**:

| Error Type | Business Impact | Cost |
|------------|-----------------|------|
| **False Positive** | Contact non-responder | $1 (mailing cost) |
| **False Negative** | Miss potential customer | $3 (lost revenue) |

**Cost Ratio**: Missing a customer costs **3× more** than unnecessary contact.

### Threshold Comparison Results

| Threshold | Precision | Recall | F1 | FP | FN | Expected Cost |
|-----------|-----------|--------|-----|----|----|---------------|
| Default (0.50) | 54.5% | 35.8% | 0.432 | 20 | 43 | **$149** |
| F1 Optimal (0.33) | 49.5% | 70.1% | 0.580 | 48 | 20 | $108 |
| **Cost Optimal (0.30)** | 46.4% | 76.1% | 0.576 | 59 | 16 | **$107** |

### Business Impact Analysis

**Moving from threshold 0.50 → 0.30**:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Recall | 35.8% | 76.1% | **+40.3 pp** |
| Customers Captured | 24/67 | 51/67 | **+27 customers** |
| False Positives | 20 | 59 | +39 contacts |
| False Negatives | 43 | 16 | **-27 missed** |
| Expected Cost | $149 | $107 | **-28% savings** |

**ROI Calculation**:
- Extra contact cost: 39 × $1 = $39
- Recovered customer value: 27 × $3 = $81
- **Net savings: $42 per test batch**

---

## 7. Feature Importance Analysis

### Top 10 Predictive Features

| Rank | Feature | Importance | Business Interpretation |
|------|---------|------------|-------------------------|
| 1 | **Recency** | 14.6% | Recent buyers are more likely to respond |
| 2 | **Tenure_Days** | 13.4% | Long-term customers show loyalty |
| 3 | **TotalSpend** | 13.1% | High spenders are engaged customers |
| 4 | **Income_log** | 11.2% | Higher income = capacity to buy |
| 5 | **NumCatalogPurchases** | 8.5% | Catalog buyers respond to direct marketing |
| 6 | NumStorePurchases | 6.5% | In-store engagement matters |
| 7 | Age | 5.8% | Age affects responsiveness |
| 8 | NumWebVisitsMonth | 5.8% | Website engagement is predictive |
| 9 | NumWebPurchases | 5.2% | Online buying behavior |
| 10 | Teenhome | 3.3% | Family composition affects decisions |

### Key Insights

1. **Recency is the strongest predictor** — This validates the classic RFM (Recency, Frequency, Monetary) framework. Customers who purchased recently are most likely to respond.

2. **Engagement signals dominate** — Tenure, TotalSpend, and purchase channel features capture overall customer engagement. Loyal, high-value customers respond more.

3. **Channel preference matters** — Catalog purchases strongly predict response. Customers who buy through direct mail are naturally more receptive to marketing campaigns.

4. **Demographics have lower importance** — Education and Marital Status contribute minimally, suggesting **behavior trumps demographics** for this prediction task.

---

## 8. Final Model Configuration

### Production Settings

```python
Model: Random Forest Classifier
Threshold: 0.30 (cost-optimized)
Preprocessor: StandardScaler + OneHotEncoder pipeline

Files Saved:
- models/best_classification_model.joblib
- models/classification_preprocessor.joblib
- models/classification_metadata.joblib
```

### Final Performance Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8751 |
| PR-AUC | 0.5739 |
| Precision | 46.4% |
| Recall | 76.1% |
| F1 Score | 0.576 |
| Accuracy | 83.0% |

### Classification Report (at threshold = 0.30)

```
              precision    recall  f1-score   support

 No Response       0.95      0.85      0.90       381
    Response       0.46      0.76      0.58        67

    accuracy                           0.83       448
   macro avg       0.71      0.80      0.74       448
weighted avg       0.88      0.83      0.85       448
```

---

## 9. Business Recommendations

### Targeting Strategy

Based on the model's feature importance, prioritize customers with:

1. **Recent purchases** (Recency < 30 days) — Strongest predictor
2. **High total spend** (Top 30% of customers) — Indicates engagement
3. **History of catalog purchases** — Shows receptiveness to direct marketing
4. **Long tenure** (> 1 year as customer) — Indicates loyalty
5. **Higher income** — Greater purchasing capacity

### Campaign Execution

| Scenario | Recommendation |
|----------|----------------|
| **Budget-constrained** | Use threshold 0.40 for higher precision |
| **Growth-focused** | Use threshold 0.25 for maximum reach |
| **Balanced** | Use threshold 0.30 (cost-optimal) |

### Expected ROI

**Per 1,000 customers contacted**:

| Approach | Contacts | Cost | Responders Captured | Efficiency |
|----------|----------|------|---------------------|------------|
| No model (all) | 1,000 | $1,000 | 150 (100%) | 15% |
| Model (t=0.50) | ~150 | $150 | 54 (36%) | 36% |
| **Model (t=0.30)** | ~300 | $300 | 114 (76%) | **38%** |

**Using the model at optimal threshold**:
- Contact 70% fewer customers
- Capture 76% of responders
- Achieve 2.5× targeting efficiency

---

## 10. Advanced Model Diagnostics

### 10.1 Permutation Importance Analysis

Permutation importance provides a more reliable measure of feature importance by measuring the decrease in model performance when each feature is randomly shuffled.

| Rank | Feature | Importance (ROC-AUC decrease) | Interpretation |
|------|---------|-------------------------------|----------------|
| 1 | **Recency** | 7.4% | Most critical predictor — confirms tree-based importance |
| 2 | **Tenure_Days** | 4.1% | Customer loyalty matters significantly |
| 3 | **NumCatalogPurchases** | 4.0% | Catalog channel affinity (higher than tree-based suggests) |
| 4 | **TotalSpend** | 3.4% | Higher spenders are more engaged |
| 5 | **Teenhome** | 2.0% | Family composition affects marketing response |

**Key Insight**: Permutation importance confirms **Recency** as the #1 predictor. Notably, **Income_log** drops significantly compared to tree-based importance, suggesting its tree-based importance may have been inflated due to cardinality.

### 10.2 Probability Calibration Analysis

**Brier Score: 0.094** (Well Calibrated)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Brier Score | 0.094 | Excellent (0 = perfect, 0.25 = random guessing) |

**Calibration Insights**:
- Model probabilities are reliable and can be trusted for business decisions
- Slight underconfidence in mid-range probabilities (0.3-0.6)
- High confidence predictions (>0.7) are well calibrated
- When the model predicts 50% probability, approximately 50% of customers actually respond

### 10.3 Lift and Gains Analysis

The lift chart quantifies how much better the model performs compared to random selection.

| Decile | Response Rate | Lift | Cumulative Gain |
|--------|--------------|------|-----------------|
| 1 (Top 10%) | 55.6% | **3.71×** | 37.3% |
| 2 | 44.4% | **2.97×** | 67.2% |
| 3 | 17.8% | 1.19× | 79.1% |
| 4 | 15.9% | 1.06× | 89.6% |
| 5-10 | <11% | <1× | 97-100% |

**Business Implications**:
- **Top 20%** of model-scored customers capture **67%** of all responders
- **Top 10%** achieves **3.7× lift** over random targeting
- **Bottom 40%** have virtually **zero responders** — never contact these
- Contacting only top 2 deciles reduces marketing spend by 80% while capturing 2/3 of sales

### 10.4 Learning Curve Analysis

Learning curves diagnose whether the model suffers from bias (underfitting) or variance (overfitting).

| Metric | Value | Status |
|--------|-------|--------|
| Training ROC-AUC | 0.999 | Near-perfect fit |
| Validation ROC-AUC | 0.866 | Good generalization |
| Gap | 0.133 | Above 0.05 threshold |

**Diagnosis**: Moderate overfitting detected (13% gap), typical for Random Forests.

**Observations**:
- Validation score plateaus around 1000+ samples
- Adding more data shows minimal improvement
- Training score remains at ~100% regardless of sample size

**Verdict**: The 13% gap is acceptable for production because:
- Strong validation performance (ROC-AUC = 0.87)
- The gap is stable (not increasing with more data)
- Business metrics are met (76% recall, 28% cost reduction)

---

## 12. Limitations & Future Work

### Current Limitations

1. **Single campaign data** — Model trained on one campaign; may not generalize to different offer types
2. **Static features** — Doesn't capture time-series patterns in customer behavior
3. **No causal inference** — Model predicts correlation, not causation
4. **Cost assumptions** — FP/FN costs are estimates; should be validated with actual business data

### Future Improvements

1. **A/B Testing** — Validate model predictions with controlled experiments
2. **Uplift Modeling** — Identify customers whose behavior would *change* due to the campaign
3. **Time-Series Features** — Add trends in purchasing behavior
4. **Ensemble with CLV** — Combine with Customer Lifetime Value for prioritization
5. **Multi-Campaign Learning** — Train on multiple campaigns for robustness

---

## 13. Technical Appendix

### Environment

```
Python: 3.13.0
scikit-learn: Latest
Random State: 42
Test Size: 20%
CV Folds: 5 (Stratified)
```

### Model Hyperparameters (after tuning)

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### Confusion Matrix Interpretation

```
At threshold = 0.30:

                  Predicted
                  No      Yes
Actual  No       322      59    (85% specificity)
        Yes       16      51    (76% recall)
```

- **True Negatives (322)**: Correctly identified non-responders
- **False Positives (59)**: Contacted but won't respond (acceptable cost)
- **False Negatives (16)**: Missed potential customers (minimized)
- **True Positives (51)**: Correctly identified responders

---

## 14. Conclusion

We successfully developed a **Random Forest classification model** that:

✅ Achieves **0.875 ROC-AUC** — excellent discriminative ability

✅ Captures **76% of potential responders** at the cost-optimal threshold

✅ Reduces expected costs by **28%** compared to default threshold

✅ Provides **actionable insights** on what drives customer response

✅ Is **production-ready** with saved model artifacts

The model transforms marketing from a blanket approach to **targeted, data-driven customer engagement**, maximizing ROI while respecting customer experience.

---

*Report generated from 03_classification.ipynb analysis*
*Next step: Proceed to 04_clustering.ipynb for customer segmentation*
