# Bank-Churn-Prediction
End-to-end churn classification on 10,000 bank customer records. Random Forest with AUC 0.87, 3-tier risk scoring, and model serialization.


# Customer Churn Prediction — Bank Dataset

Predicts whether a bank customer will churn using supervised classification.
Covers the full pipeline: cleaning, feature engineering, cross-validation, evaluation, and model saving.

---

## Dataset

- **Source:** Bank customer records (Kaggle)
- **Size:** 10,000 rows, 18 columns
- **Target:** `Exited` — 1 if the customer churned, 0 if they stayed
- **Class split:** ~20% churned, ~80% stayed

---

## What was removed and why

| Column | Reason |
|---|---|
| `RowNumber`, `CustomerId`, `Surname` | Identifiers — no predictive value |
| `Complain`, `Satisfaction Score` | Data leakage — only known after churn occurs |

---

## Feature Engineering

- **`ZeroBalance`** — binary flag for customers with a zero account balance (distinct behavioural pattern)
- All categorical columns (`Geography`, `Gender`, `Card Type`) one-hot encoded with `pd.get_dummies`

---

## Models

| Model | CV AUC | Test AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | — | **0.8704** | 0.5941 | 0.8128 | 0.4681 | 86.95% |
| Logistic Regression | — | 0.7776 | 0.3144 | 0.6187 | 0.2108 | 81.25% |

Random Forest selected as best model based on AUC.

---

## Validation

- **Stratified 5-Fold Cross-Validation** on training data to check stability
- StandardScaler applied only to Logistic Regression (tree models don't need it)
- 80/20 stratified train-test split — churn rate preserved in both sets

---

## Risk Scoring

Churn probabilities from the best model are mapped to a 3-tier risk system:

| Tier | Probability |
|---|---|
| Low | < 0.45 |
| Medium | 0.45 – 0.69 |
| High | ≥ 0.70 |

---

## Project Structure

```
customer-churn-prediction/
├── customer_churn.ipynb        # full pipeline notebook
├── Customer-Churn-Records.csv  # dataset
└── saved_models/
    ├── Random_Forest.pkl
    ├── Logistic_Regression.pkl
    ├── scaler.pkl
    ├── feature_names.json
    └── model_results.csv
```

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Open `customer_churn.ipynb` in Jupyter Notebook and run all cells top to bottom.

---

## Key Takeaways

- High precision (81%) means when the model flags a customer as at-risk, it is usually right — useful for targeted retention campaigns
- Recall is lower (47%) — some churners are missed, which is expected given the class imbalance
- Age and account balance were the strongest predictors per Random Forest feature importances
