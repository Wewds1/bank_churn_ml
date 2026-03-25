# Bank Churn Prediction Model Card

## Model Overview
- **Name:** ExtraTrees Classifier
- **Type:** ExtraTreesClassifier  
- **Training Date:** March 25, 2025
- **Purpose:** Identify at-risk customers for targeted retention campaigns

## Performance Warning
**Current threshold (0.34) prioritizes precision over recall, catching only 6% of churners.**

## Final Test Metrics (Threshold: 0.34)
- **ROC-AUC:** 0.7083
- **PR-AUC:** 0.1169  
- **Recall:** 0.0606 (6% of churners detected)
- **Precision:** 0.2000 (20% of predictions are correct)
- **F1-Score:** 0.0930

## Data Splits
- **Training:** 6,179 samples
- **Validation:** 2,060 samples
- **Test:** 2,060 samples
- **Class Distribution:** ~1.6% churn (severely imbalanced)

## Key Features (Top 5)
- account_age_days
- balance_income_ratio
- txn_per_product
- complaint_txn_ratio
- low_engagement_flag

## Known Limitations
- Low recall suggests threshold may need tuning for production
- Relies on historical transaction patterns; sudden behavior changes not captured
- Model trained on 2026 Q1 data; may degrade on future cohorts
- Imbalanced training may limit minority class detection

## Recommended Next Steps
- **Immediate:** Reduce threshold to 0.10–0.15 to improve recall
- Re-evaluate validation threshold optimization to balance recall/precision
- Verify if another model (GradientBoosting, RandomForest) performed better in CV