# Bank Churn ML Pipeline

An end-to-end machine learning workflow for predicting customer churn in a retail banking dataset.
The goal is to flag at-risk customers early enough for targeted retention.

---

## Problem context

**Target variable:** `churned`

The dataset is heavily imbalanced and ships with several real-world data issues that make it a useful exercise beyond a clean Kaggle CSV:

- Leakage columns that inflate metrics if left in
- `annual_income` stored in mixed text formats
- Missing values with different underlying mechanisms (MCAR / MAR / MNAR)
- Duplicate customer records that can contaminate a train/test split
- A mix of numeric and categorical features requiring separate preprocessing paths

---

## Repository layout

```
bank_churn_raw.csv          # Original source data — never modified
bank_churn_cleaned.csv      # Output of the cleaning stage
01_eda.ipynb                # Exploratory analysis and diagnostic findings
02_data_cleaning.ipynb      # Deterministic cleaning pipeline
03_modeling.ipynb           # Feature engineering, training, and evaluation
```

---

## Workflow

### 1 — EDA (`01_eda.ipynb`)

- Shape, dtypes, null profile, target distribution
- Leakage diagnosis for three columns:
  - `exit_survey_score`
  - `account_closed_date`
  - `churn_flag_internal`
- Missingness screening with heuristic MCAR / MAR / MNAR classification
- Decision table summarizing what to drop, impute, or flag

### 2 — Data cleaning (`02_data_cleaning.ipynb`)

- Drop leakage columns
- Parse `annual_income` into `annual_income_cleaned` (mixed text → numeric)
- Deduplicate on `customer_id`
- Add binary missing indicators for high-risk fields
- Apply numeric and categorical imputation
- Save output to `bank_churn_cleaned.csv`

### 3 — Modeling (`03_modeling.ipynb`)

- Load cleaned data
- Engineer additional behavioral features
- Stratified train/test split
- `ColumnTransformer` + `Pipeline` for preprocessing
- Two baseline models:
  - Logistic Regression with `class_weight='balanced'`
  - Random Forest with balanced subsampling
- Evaluation metrics: ROC-AUC, PR-AUC, Recall, Precision, F1, confusion matrix
- ROC and Precision-Recall curves
- Threshold tradeoff analysis for churn targeting decisions

---

## Design decisions

| Decision | Reason |
|---|---|
| Drop leakage columns before any modeling | Prevents post-outcome information from inflating metrics |
| Parse `annual_income` to numeric | Mixed text formats are unusable as a raw feature |
| Deduplicate before the train/test split | Identical records in both sets contaminate evaluation |
| Missing indicators for high-risk columns | Preserves the signal in missingness when it is informative |
| `Pipeline` + `ColumnTransformer` | Guarantees consistent transformations; reduces preprocessing leakage |
| Imbalance-aware metrics | Accuracy alone is misleading for rare-event prediction |

---

## Known limitations and next steps

- [ ] Threshold tuning and model selection should use a held-out validation set, not the test set
- [ ] Add cross-validated model comparison
- [ ] Add calibration checks for predicted probabilities
- [ ] Persist trained model and preprocessing artifacts under a `models/` directory
- [ ] Add unit tests for `parse_income` and key cleaning assumptions

---

## Reproducing results

1. Keep `bank_churn_raw.csv` immutable — treat it as the source of truth
2. Run `02_data_cleaning.ipynb` to regenerate `bank_churn_cleaned.csv`
3. Run `03_modeling.ipynb` to train and evaluate models

All notebooks use fixed `random_state` values for deterministic output.

---

## Summary

This project covers a realistic ML engineering workflow from messy tabular data to baseline churn prediction models — with explicit handling of leakage, missingness, class imbalance, and threshold-based decisioning.