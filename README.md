# Bank Churn ML Pipeline

An end-to-end machine learning workflow for predicting customer churn in a retail banking dataset.
The goal is to flag at-risk customers early enough for targeted retention.

---

## Repository layout

```
data/
├── raw/                        # Original source data — never modified
│   └── bank_churn_raw.csv
├── interim/                    # After cleaning, before feature engineering
│   └── bank_churn_cleaned.csv
└── processed/                  # Final train/val/test splits
    ├── X_train.csv, X_val.csv, X_test.csv
    └── y_train.csv, y_val.csv, y_test.csv

notebooks/
├── 01_eda.ipynb                # Exploratory analysis and diagnostics
├── 02_data_cleaning.ipynb      # Deterministic cleaning pipeline
└── 03_modeling.ipynb           # Feature engineering, training, evaluation

src/
├── __init__.py
├── data_prep.py                # Parsing, leakage/duplicate detection
├── features.py                 # Feature engineering functions
└── train.py                    # Pipeline building utilities

models/
└── best_churn_model.pkl        # Serialized trained ExtraTreesClassifier

reports/
├── model_card.md               # Model documentation and limitations
└── figures/                    # Evaluation plots and explainability
    ├── 01_roc_curve.png
    ├── 02_pr_curve.png
    ├── 03_confusion_matrix.png
    ├── 04_feature_importance.png
    └── 05_shap_summary.png

tests/
├── test_data_prep.py           # Unit tests for cleaning functions
└── testing.ipynb               # Sandbox notebook for exploration

requirements.txt                # Pinned dependencies
.gitignore
README.md
```

---

## Workflow

### 1 — EDA (`01_eda.ipynb`)

- Shape, dtypes, null profile, target distribution
- Leakage diagnosis: identifies columns that encode the target
- Missingness screening: MCAR vs MNAR classification
- Decision table: what to drop, impute, or flag

**Output:** Diagnostic tables; informs cleaning strategy

### 2 — Data cleaning (`02_data_cleaning.ipynb`)

- Drop leakage columns
- Parse `annual_income` (mixed text → numeric)
- Deduplicate on `customer_id`
- Add binary missing indicators for high-risk fields
- Apply numeric and categorical imputation

**Output:** `data/interim/bank_churn_cleaned.csv`

### 3 — Modeling (`03_modeling.ipynb`)

- Load cleaned data
- Engineer behavioral features (ratios, flags, composites)
- Stratified train/val/test split (retains class balance)
- Build preprocessing pipeline (imputation + scaling + one-hot encoding)
- 5-fold cross-validation across 5 candidate models:
  - Logistic Regression (baseline)
  - Random Forest
  - ExtraTrees — winner (CV PR-AUC: 0.1532)
  - Gradient Boosting
  - Gaussian Naive Bayes
- Threshold tuning on validation set
- Final evaluation on held-out test set
- Generate ROC, PR, confusion matrix, feature importance, and SHAP plots
- Serialize model to `models/best_churn_model.pkl`
- Export processed splits for reproducibility

---

## Final model performance

**Model:** ExtraTreesClassifier (500 trees, balanced class weights)  
**Decision threshold (tuned on validation set):** 0.34

### Test set metrics

| Metric | Value |
|---|---|
| ROC-AUC | 0.7083 |
| PR-AUC | 0.1169 |
| Recall | 0.0606 |
| Precision | 0.20 |
| F1 | 0.093 |

### Confusion matrix (test set)

|  | Predicted: No churn | Predicted: Churn |
|---|---|---|
| Actual: No churn | 2019 | 8 |
| Actual: Churn | 31 | 2 |

**Note:** Low recall reflects the extreme class imbalance (1.6% churn rate). The model is conservative — it avoids false alarms but misses most actual churners. For a production retention campaign, consider lowering the threshold to 0.15–0.25 to trade precision for recall.

---

## Design decisions

| Decision | Reason |
|---|---|
| Drop leakage columns before any modeling | Prevents post-outcome information from inflating metrics |
| Parse `annual_income` to numeric | Mixed text formats are unusable as raw features |
| Deduplicate before train/test split | Identical records in both sets contaminate evaluation |
| Missing indicators for high-risk columns | Preserves the signal in missingness (MNAR) |
| `Pipeline` + `ColumnTransformer` | Guarantees consistent transformations; prevents preprocessing leakage |
| Imbalance-aware metrics (PR-AUC over ROC-AUC) | Accuracy alone is misleading for rare-event prediction |
| ExtraTrees as final model | Best CV PR-AUC; tree-based structure supports interpretability |

---

## Reproducing results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run cleaning pipeline
jupyter notebook notebooks/02_data_cleaning.ipynb

# 3. Run modeling pipeline
jupyter notebook notebooks/03_modeling.ipynb
```

All notebooks use fixed `random_state` values for deterministic output. Keep `data/raw/bank_churn_raw.csv` immutable — treat it as the source of truth.

---

## Known limitations and next steps

- [ ] Add calibration checks for predicted probabilities
- [ ] Expand unit tests to cover feature engineering functions
- [ ] Document threshold sensitivity analysis in `model_card.md`
- [ ] Explore SMOTE or cost-sensitive learning to improve recall