# 🎯 A Personalized Recommendation System for Behavioral-Based Financial Planning
---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Machine Learning Model](#machine-learning-model)
- [Financial Behavior Personas](#financial-behavior-personas)
- [Dataset](#dataset)
- [Features](#features)
- [How It Differs from Excel](#how-it-differs-from-excel)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Goal Feasibility Analysis](#goal-feasibility-analysis)
- [Testing & Results](#testing--results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## 📌 Overview

Most existing financial planning tools — like Mint, YNAB, and PocketGuard — provide **generic recommendations** that treat all users the same, regardless of their individual spending habits, saving tendencies, or financial goals. This means users receive advice that does not reflect their actual financial behavior, leading to unrealistic budgets, poor adherence, and continued financial stress.

This project addresses that gap by building a **personalized recommendation system** that:

1. Analyzes individual financial behavior from user data
2. Classifies users into distinct **financial behavior personas** using machine learning
3. Assesses whether a user's financial goal is **realistically achievable**
4. Generates **personalized, behavior-aware financial recommendations** based on the above

---

## ❗ Problem Statement

Many individuals struggle with personal financial management despite having access to digital financial tools. The core problem is that existing tools:

- Offer **generic advice** that ignores individual behavioral patterns
- Do not analyze behavioral drivers like impulse spending, emotional spending, or subscription overload
- Provide **no goal feasibility analysis** — no assessment of whether a user can actually achieve their stated goal
- Cannot classify users based on their **unique financial habits**

This leads to unrealistic budgets, low adherence, poor saving culture, and financial stress.

---

## ⚙️ How It Works

The system follows a **6-step modular pipeline**:

```
User Input → Data Preprocessing & Feature Engineering → Behavioral Clustering
    → Goal Feasibility Analysis → Recommendation Engine → Output & Visualization
```

### Step-by-Step:

1. **User inputs** their financial data: income, expenses (housing, food, transport, entertainment, subscriptions), savings, debt, financial goal, and goal timeline
2. **Feature engineering** computes behavioral indicators from the raw data
3. **K-Means clustering** classifies the user into one of four financial behavior personas
4. **Goal feasibility analysis** checks if the user's goal is achievable given their savings capacity
5. **Rule-based recommendation engine** generates personalized advice based on the persona + feasibility result
6. **Output module** displays the user's financial profile, persona, feasibility result, and recommendations visually

---

## 🏗️ System Architecture

The system is built around **six independent but connected modules**:

| Module | Responsibility |
|---|---|
| User Input Module | Collects and validates raw financial data |
| Data Preprocessing & Feature Engineering | Cleans data, computes behavioral indicators |
| Behavioral Clustering Module | Applies K-Means to classify user into a persona |
| Goal Feasibility Analysis Module | Assesses if financial goal is achievable |
| Recommendation Engine | Generates persona-specific personalized recommendations |
| Output & Visualization Module | Displays results through charts and text |

Each module operates independently but communicates through a centralized data processing pipeline, ensuring **flexibility, maintainability, and ease of future enhancement**.

---

## 🛠️ Tech Stack

### Frontend / UI
| Technology | Purpose |
|---|---|
| **React 19** | UI framework — component-based interface |
| **TypeScript** | Programming language — static typing for reliability |
| **Vite** | Build tool and dev server — fast HMR during development |
| **React Router DOM v7** | Client-side navigation between views |
| **Recharts** | Interactive charts and data visualizations |
| **Lucide React** | UI icons |
| **clsx** | Conditional CSS class management |
| **npm** | Package manager |

### Machine Learning & Data Analysis
| Technology | Purpose |
|---|---|
| **Python** | Core language for ML pipeline |
| **Pandas & NumPy** | Data manipulation and preprocessing |
| **Scikit-learn** | K-Means clustering, StandardScaler, PCA |
| **Matplotlib & Seaborn** | EDA visualizations |

---

## 🤖 Machine Learning Model

### Model: K-Means Clustering (Unsupervised Learning)

**Why K-Means?**

- The goal was to **discover** behavioral groupings from financial data — not predict a predefined label. This is a clustering/grouping task, which unsupervised learning is specifically designed for.
- K-Means is **computationally efficient** for structured numerical data like financial records.
- It produces **interpretable cluster centroids** that can be meaningfully labeled as financial personas.
- The **Elbow Method** could be used to objectively determine the optimal number of clusters.

**Why Unsupervised and not Supervised?**

The dataset contained financial figures (income, expenses, savings etc.) but carried **no predefined behavioral labels**. No record was pre-tagged as "High Spender" or "Balanced Saver." The persona classifications were **discovered by the model itself** from patterns in the data — the labels were only assigned *after* clustering, based on analysis of each cluster's centroid values.

**How overfitting was avoided:**

- K-Means is naturally less prone to overfitting than supervised models since there are no labeled outputs to memorize
- The **Elbow Method** prevented over-clustering by identifying k=4 as the point of diminishing returns
- **StandardScaler normalization** ensured no single feature dominated the model due to scale
- Only **meaningful derived behavioral features** were used, avoiding noise

### Feature Engineering

The following behavioral indicators were computed from raw financial data:

| Feature | Formula |
|---|---|
| Savings Rate | Monthly Savings ÷ Monthly Income |
| Discretionary Spending Ratio | (Entertainment + Subscriptions) ÷ Monthly Income |
| Expense-to-Income Ratio | Total Monthly Expenses ÷ Monthly Income |
| Debt-to-Income Ratio (DTI) | Total Debt ÷ (Monthly Income × 12) |
| Required Monthly Savings | Goal Amount ÷ Goal Timeline |
| Goal Feasibility Score | Required Monthly Savings ÷ Actual Monthly Savings |

### Why Four Clusters?

The number of clusters (k=4) was **not chosen arbitrarily**. The **Elbow Method** was applied by plotting the Within-Cluster Sum of Squares (WCSS) against different values of k. At k=4, the graph produced a clear "elbow" — meaning adding more clusters beyond 4 produced diminishing returns in cluster quality.

The 4 clusters were also validated by their **behavioral interpretability** — each produced a distinct and meaningful financial persona.

---

## 👤 Financial Behavior Personas

| Persona | Behavioral Profile | Recommendation Focus |
|---|---|---|
| **Balanced Saver** | Healthy savings rate, moderate debt, controlled discretionary spending | Goal acceleration, investment readiness, maintaining positive habits |
| **High Spender** | High discretionary expenditure, low savings rate, moderate-high debt | Discretionary spending reduction, subscription audit, savings automation |
| **Debt-Burdened** | High DTI ratio, low savings, limited discretionary flexibility | Debt repayment prioritization, emergency fund building, expense optimization |
| **Goal-Oriented** | High savings rate, low discretionary spending, focused on goal achievement | Goal timeline optimization, milestone tracking, savings maintenance |

---

## 📊 Dataset

### Type: Synthetic (Simulated) Data

**Why synthetic data was used:**

1. **Privacy & Ethics** — Real financial data is highly sensitive. Collecting it would have required ethical clearance, informed consent, and compliance with Kenya's Data Protection Act (2019) — beyond the scope of an undergraduate project.
2. **Accessibility** — Real financial data from banks or M-Pesa is not publicly available and would have required formal institutional partnerships.
3. **Control** — Synthetic data allowed deliberate representation of a wide range of financial behaviors across different income levels, spending patterns, and debt situations.
4. **Proof of Concept** — The primary goal was to demonstrate the methodology works in principle. Synthetic data was sufficient for this purpose.

### Dataset Structure

| Field | Type | Description |
|---|---|---|
| Income | Float | Monthly gross income (KSh) |
| Housing | Float | Monthly rent/housing expenditure |
| Food | Float | Monthly food expenditure |
| Transport | Float | Monthly transport expenditure |
| Entertainment | Float | Monthly entertainment expenditure |
| Subscriptions | Float | Monthly subscription services |
| Savings | Float | Monthly savings amount |
| Debt | Float | Total outstanding debt |
| Goal Amount | Float | Financial goal target (KSh) |
| Goal Timeline | Integer | Months to achieve goal |

### Sample Records

| Income | Housing | Food | Transport | Entertainment | Subscriptions | Savings | Debt | Goal Amt | Timeline |
|---|---|---|---|---|---|---|---|---|---|
| 60,000 | 20,000 | 10,000 | 6,000 | 8,000 | 3,000 | 5,000 | 30,000 | 50,000 | 10 mo |
| 45,000 | 18,000 | 9,000 | 5,000 | 4,000 | 2,000 | 3,000 | 20,000 | 30,000 | 12 mo |
| 80,000 | 25,000 | 12,000 | 7,000 | 15,000 | 5,000 | 10,000 | 40,000 | 100,000 | 12 mo |

> **Dataset size:** 100 records | **Currency:** Kenyan Shillings (KSh)

---

## ✨ Features

- 📥 **Financial Data Input** — Users input income, expenses across 6 categories, savings, debt, and financial goal details
- 🧮 **Automated Feature Engineering** — System computes savings rate, DTI ratio, discretionary spending ratio automatically
- 🧠 **Behavioral Persona Classification** — K-Means clustering assigns user to one of 4 financial personas
- 🎯 **Goal Feasibility Analysis** — System assesses whether the user's financial goal is Feasible, Partially Feasible, or Not Feasible
- 💡 **Personalized Recommendations** — 3–5 targeted recommendations generated per user based on persona + feasibility
- 📊 **Visual Financial Dashboard** — Bar charts and visualizations for spending distribution and savings rates built with Recharts
- 🔀 **Multi-view Navigation** — React Router DOM enables seamless navigation between system views

---

## 📊 How It Differs from Excel

| Feature | Excel | This System |
|---|---|---|
| Behavioral Intelligence | ❌ None | ✅ K-Means clustering identifies behavioral patterns |
| Personalized Recommendations | ❌ None — you interpret numbers yourself | ✅ Auto-generated recommendations per persona |
| Goal Feasibility Analysis | ❌ Manual calculation | ✅ Automated feasibility assessment |
| Machine Learning | ❌ Not capable | ✅ Scikit-learn K-Means clustering |
| Automation | ❌ Manual entry and formula writing | ✅ Full pipeline from input to recommendations |
| Scalability | ❌ Struggles with many users | ✅ Modular architecture supports multiple profiles |

**In short — Excel shows you numbers. This system understands your behavior and tells you what to do about it.**

---

## 🔍 Exploratory Data Analysis

EDA was conducted using **Pandas, NumPy, Matplotlib, and Seaborn** and included:

- **Descriptive Statistical Analysis** — Mean, median, standard deviation of income, expenses, savings, and debt
- **Spending Pattern Analysis** — Distribution of income across expense categories
- **Feature Distribution Visualization** — Distributions of savings rate, DTI ratio, and discretionary spending ratio
- **Correlation Analysis** — Relationships between financial variables and savings/debt behavior
- **PCA Visualization** — 2D PCA projection to visually validate cluster separation after K-Means

---

## 🎯 Goal Feasibility Analysis

Users were categorized into three feasibility groups:

| Category | Condition |
|---|---|
| **Feasible** | Required monthly savings ≤ current monthly savings |
| **Partially Feasible** | Required monthly savings between 100%–150% of current savings |
| **Not Feasible** | Required monthly savings > 150% of current savings |

---

## ✅ Testing & Results

### Test Cases

| TC# | Test Case | Result |
|---|---|---|
| TC01 | Data input validation | ✅ Pass |
| TC02 | Feature computation accuracy | ✅ Pass |
| TC03 | Clustering — Balanced Saver profile | ✅ Pass |
| TC04 | Clustering — High Spender profile | ✅ Pass |
| TC05 | Goal feasibility — Feasible | ✅ Pass |
| TC06 | Goal feasibility — Not Feasible | ✅ Pass |
| TC07 | Recommendation generation | ✅ Pass |
| TC08 | UAT — Usability | ✅ Pass |

### UAT Results (5 pilot users)

| Question | Result |
|---|---|
| Found system easy to use | 100% |
| Persona accurately reflected financial habits | 80% |
| Recommendations were actionable | 100% |

---

## ⚠️ Limitations

1. **Synthetic data** — Does not capture all nuances of real financial behavior
2. **100 records** — Small dataset; larger more diverse data would improve cluster robustness
3. **No database** — Data exists only in session memory; lost on page refresh. No user accounts or persistent storage
4. **No real-time data** — No integration with banking APIs or M-Pesa
5. **Static recommendations** — Rule-based engine does not adapt over time based on behavior changes
6. **Proof of concept** — Not a production-ready system

---

## 🚀 Future Improvements

1. **Real-World Data Integration** — Connect to Safaricom Daraja API (M-Pesa) for real transaction data
2. **Database** — Add Firebase (Firestore) for data persistence and user authentication, chosen for its seamless React integration and real-time capabilities
3. **Advanced ML Models** — Explore supervised classification and reinforcement learning for adaptive recommendations
4. **Longitudinal Behavioral Tracking** — Track behavior changes over time, enable adaptive persona reassignment
5. **Expanded Persona Granularity** — Refine personas using larger, more demographically diverse datasets
6. **Mobile Application** — Build a mobile-first version for smartphone accessibility
7. **Financial Literacy Content** — Pair recommendations with curated educational resources
8. **Regulatory Compliance** — Comply with Kenya's Data Protection Act (2019) for any commercial deployment

---

## 📚 References

- Ahmed, S., Patel, R., & Mehta, K. (2021). Personalized financial recommendation systems: A review. *Journal of Financial Technology Research, 5*(2), 45–60.
- García, M., Lee, J., & Patel, K. (2020). Mobile applications for personal financial management. *Journal of Digital Finance, 4*(3), 22–38.
- Johnson, T. (2021). Evaluating financial apps for personal budgeting. *International Journal of Digital Finance, 3*(1), 15–29.
- Kumar, A., & Lee, J. (2019). Limitations of generic personal finance tools. *Journal of Consumer Finance, 7*(3), 50–62.
- Peiris, T. U. I. (2021). Effect of financial literacy on individual savings behavior. *European Journal of Business and Management Research, 6*(5), 1–6.
- Sarah, K. C., & Mule, R. K. (2023). Do budgeting practices affect saving behavior among small-scale entrepreneurs in Kenya? *European Scientific Journal, 19*(1), 87–102.
- Smith, L., & Brown, R. (2020). Zero-based budgeting applications. *Journal of Personal Finance, 12*(4), 88–102.
- Taylor, P. (2021). Educating users through budgeting apps. *Financial Literacy Review, 8*(2), 33–47.
- Thaler, R. H., & Sunstein, C. R. (2008). *Nudge.* Yale University Press.
- Wamukota, M., & Luutsa, F. E. (2025). Impact of financial literacy on expenditure, savings, and investments. *Strategic Journal of Business & Change Management, 12*(1), 330–342.

---

> *This project was submitted in partial fulfillment of the requirements for the award of the Bachelor of Science in Data Science at The Co-operative University of Kenya, January 2026.*
