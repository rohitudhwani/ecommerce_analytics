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

## Model Performance

| Model | ROC-AUC | Accuracy |
|---|---|---|
| XGBoost (final) | **0.817** | 75% |
| Logistic Regression (baseline) | 0.808 | 75% |
| 5-Fold CV Mean AUC | **0.816 ± 0.008** | — |

The near-identical accuracy between XGBoost and Logistic Regression is intentional and informative — it confirms the features themselves carry most of the predictive signal. The model is **stable** (low CV variance) and **consistent** across both approaches.

---

## Dataset

**UCI Online Retail II** — Real UK-based non-store online retail transactions, Dec 2009 – Dec 2011.

- 1M+ raw transaction rows across two years
- Mainly gift-ware sold to wholesale buyers
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

**Churn Definition:** A customer is labelled as churned if they made **zero purchases in the 90-day prediction window** following the observation cutoff (2011-09-10). This mirrors how production churn models are built in industry.

---

## Feature Engineering

12 features engineered from raw transaction data — no external data required.

| Feature | Description | Churn Signal |
|---|---|---|
| **Recency** | Days since last purchase | ↑ High recency → churn risk (SHAP: 0.833) |
| **CancellationRate** | % of transactions cancelled | ↑ High rate → churn risk (SHAP: 0.310) |
| **AvgDaysBetweenOrders** | Purchase rhythm (days) | ↑ Irregular buyers churn more |
| **UniqueProducts** | Breadth of product engagement | ↓ Wider buyers are stickier |
| **Monetary** | Total spend in observation window | Context for absence signal |
| **AOV** | Average order value | Basket behaviour indicator |
| **Frequency** | Number of unique orders | ↓ More orders = lower churn |
| **IsQ4Buyer** | Purchased in Sep–Dec (flag) | ↓ Seasonal wholesale buyers are loyal |
| **Tenure** | Days between first and last purchase | ↓ Longer tenure = lower churn |
| **AvgItemsPerOrder** | Order basket size | Purchasing intensity |
| **IsHighValue** | Top 5% order value customers (flag) | ↓ High-value buyers churn less |
| **IsUK** | UK-based customer (flag) | Slight positive churn signal |

---

## SHAP Explainability

Model decisions are fully explainable using SHAP (SHapley Additive exPlanations), making results trustworthy and auditable.

**Top insight from the beeswarm plot:**
- Customers with **high recency** (red dots, long time since purchase) are pushed strongly toward churn — this is the dominant signal by a wide margin
- Customers with **high UniqueProducts** (red dots, wide product range) are pulled toward retention — product breadth is a proxy for engagement and switching cost
- **CancellationRate** is a dissatisfaction signal: high cancellers leave

**Two real customers explained by the model:**

| | Highest Risk Customer | Safest Customer |
|---|---|---|
| Churn Probability | **99.1%** | **0.6%** |
| Recency | 639 days | 3 days |
| Frequency | 2 orders | 50 orders |
| Cancellation Rate | 63.1% | 10.5% |
| Unique Products | 24 | 583 |
| Tenure | 0 days | 644 days |
| Interpretation | Bought twice, cancelled most of it, vanished for 2 years | Wholesale account buying every 13 days across 583 products |

---

## Customer Risk Tiers

| Risk Tier | Customers | Avg Churn Prob | Revenue at Risk |
|---|---|---|---|
| 🔴 High Risk | 1,964 | 88% | £1,446,688 |
| 🟠 Medium Risk | 1,234 | 63% | £955,108 |
| 🔵 Low Risk | 947 | 37% | £609,315 |
| 🟢 Safe | 1,136 | 11% | £1,042,866 |

---

## Retention Action Recommendations

| Priority | Segment | Signal | Recommended Action | Revenue Protected |
|---|---|---|---|---|
| 🔴 Immediate | High Risk (1,964 customers) | Churn prob ≥ 75%, high recency | Win-back email + 10% discount voucher | £1,446,688 |
| 🔴 Immediate | High-Value at Risk (137 customers) | High-value + medium/high risk tier | Personal account manager outreach + loyalty incentive | £954,782 |
| 🟠 This Month | Medium Risk (1,234 customers) | Churn prob 50–75%, irregular cadence | Re-engagement newsletter + product recommendations | £955,108 |
| 🟠 This Month | One-Time Buyers (1,576 customers) | Frequency = 1, churn rate 79.8% | Second-purchase campaign: "Complete your collection" | £428,983 |
| 🟡 Next Quarter | Non-Q4 Buyers (1,843 customers) | Churn rate 72.2%, no seasonal purchase | Off-season catalogue drop — target Jan/Feb with spring range | £1,458,620 |

> **The single highest-leverage action:** Converting one-time buyers to repeat buyers. At 79.8% churn, this group represents a structural leak. A targeted second-purchase campaign with a modest incentive could realistically recover a significant portion of the £428K at risk — and permanently shift those customers into the repeat buyer segment (46.7% churn rate).

---

## Key EDA Insights

**Seasonality:** Revenue and order volume nearly double in Q4 (Sep–Dec) across both years. This is wholesale Christmas stocking behaviour, not retail impulse buying. The spike is in *volume*, not basket size — average order value does not increase in Q4.

**Product concentration:** The top 2 products (Regency Cakestand 3 Tier at £286K, White Hanging Heart T-Light Holder at £252K) together account for over £500K in revenue. High product concentration is a supply chain risk worth flagging.

**High-value segment:** 521 customers (top 5% by order value) drive **33.2% of total revenue** — £5.9M. The 95th percentile order threshold is only £1,254, suggesting these are consistent wholesale buyers rather than one-off large orders.

**Volume vs. margin:** High-volume products are predominantly low unit price. The highest-revenue products are moderate volume at moderate-to-high price. Pure volume is not a reliable proxy for revenue contribution.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Baseline model, preprocessing, evaluation |
| XGBoost | Final churn classifier |
| SHAP | Model explainability |
| matplotlib, seaborn | Visualisation |
| Google Colab | Development environment |

---

## How to Run

1. Download the dataset from [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
2. Upload `online_retail_II.xlsx` to your Colab environment (or Jupyter if you prefer)
3. Open `notebooks/churn_analysis.ipynb`
4. Run all cells sequentially — each section is self-contained and labelled

---

## Author

**Rohit Udhwani**
Data Analyst (trying to become a Data Scientist)
[LinkedIn](https://www.linkedin.com/in/rohitudhwani/) · [GitHub](https://github.com/rohitudhwani)

> *This project was built as part of a portfolio transition from Data Analysis to Data Science. The focus was on business-first framing: every modelling decision is tied to a commercial outcome, and every output is designed to be actionable by a non-technical stakeholder.*

---

## License

Dataset: UCI Machine Learning Repository (public domain)
Code: MIT License
