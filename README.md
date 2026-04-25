# Customer Churn Prediction & Revenue Recovery Engine
### eCommerce | Machine Learning | XGBoost | SHAP Explainability

---

## Project Overview

This project builds an end-to-end **customer churn prediction system** on a real-world eCommerce dataset. It identifies customers at risk of churning, explains *why* they are at risk using SHAP explainability, quantifies the **revenue at stake**, and delivers prioritised retention recommendations — the kind of output a business can act on immediately.

> **Business framing:** Every eCommerce company loses customers silently. This system makes churn visible, explainable, and preventable — before revenue walks out the door.

---

## Key Business Findings

| Metric | Value |
|---|---|
| Customers analysed | 5,281 |
| Total historical revenue | £14,287,408 |
| Overall churn rate | 56.6% |
| **Revenue at risk (High + Medium tier)** | **£2,401,796** |
| High-risk customers | 1,964 |
| High-value customers at risk | 137 |
| One-time buyer churn rate | 79.8% |
| High-value customer churn rate | 30.9% |

---

## 🎯 Model Performance

| Model | ROC-AUC | Accuracy |
|---|---|---|
| XGBoost (final) | **0.817** | 75% |
| Logistic Regression (baseline) | 0.808 | 75% |
| 5-Fold CV Mean AUC | **0.816 ± 0.008** | — |

The near-identical accuracy between XGBoost and Logistic Regression is intentional and informative — it confirms the features themselves carry most of the predictive signal. The model is **stable** (low CV variance) and **consistent** across both approaches.

---

