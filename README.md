<div align="center">

# 📉 Telco Customer Churn Analysis

Predicting _which_ customers will leave — and _why_ — with an end-to-end machine learning pipeline

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-Boosting-189FDD?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/>
</p>

</div>

<br>

## 🎯 Overview

Customer **churn** — the rate at which customers stop doing business with a company — is one of the most expensive problems a subscription business faces. This project investigates churn for a fictional telecom provider (Telco Co.) using **~7,043 customers across 21 attributes**, and answers two questions every business cares about:

> **1. Can we predict who is about to churn?**<br>
> **2. What are the biggest drivers behind it?**

The work follows the industry-standard **CRISP-DM** process — from business understanding and data cleaning through statistical analysis, visualization, and the training/evaluation of **7 machine-learning models** — all delivered in a single, thoroughly documented notebook.

<br>

## 🏆 Key Highlights

- 🎯 **Best prediction accuracy** — **~81%** (Logistic Regression / XGBoost).
- 🔑 **Top churn drivers** — Month-to-Month contract · No Online Security · No Tech Support.
- 🤖 **Models trained & compared** — 7 (Logistic Regression, SVM, kNN, Naïve Bayes, Random Forest, AdaBoost, Gradient Boosting / XGBoost).
- 🔬 **Feature combinations searched** — every unique combination of **1 → 18 features**.
- 📊 **Dataset** — 7,043 customers · 21 features.

**Bottom line:** customers on month-to-month contracts, with short tenures and higher monthly charges, and without security/support add-ons are the most likely to churn — and a tuned model flags them with ~81% accuracy.

<br>

## 🛠️ Tech Stack

| Layer                | Technologies             |
| :------------------- | :----------------------- |
| **Language**         | Python 3.10              |
| **Environment**      | Jupyter Notebook         |
| **Data**             | pandas, NumPy, itertools |
| **Visualization**    | Matplotlib, Seaborn      |
| **Machine Learning** | scikit-learn             |
| **Boosting**         | XGBoost                  |

<br>

## ✨ Features & Methodology

### 🧹 Data Preparation & Cleaning

- Type correction (`TotalCharges` object → float) and binary encoding of the `Churn` target.
- Missing-value detection & removal, redundant-data detection, descriptive statistics.

### 📈 Exploratory & Statistical Analysis

- **Variance & standard deviation** analysis to understand data spread.
- **Correlation analysis** focused on churn drivers.
- **Histogram analysis** of every variable, including churn-segmented comparisons and skewness classification.
- **Box-plot analysis** of tenure and charges against churn.

### 🤖 Predictive Modelling

Seven classifiers trained, tuned, and benchmarked — each reporting accuracy, correct/incorrect counts, confusion matrices, and feature-importance insights:

| Family          | Models                                                                       |
| :-------------- | :--------------------------------------------------------------------------- |
| Linear          | Logistic Regression                                                          |
| Distance-based  | k-Nearest Neighbors _(k-fold CV tuned)_                                      |
| Probabilistic   | Naïve Bayes (Bernoulli)                                                      |
| Kernel          | Support Vector Machine                                                       |
| Tree / Ensemble | Random Forest, AdaBoost, Gradient Boosting, XGBoost _(`GridSearchCV` tuned)_ |

### 🔍 Exhaustive Feature-Combination Search

Reusable training functions systematically train models on **every unique combination of 1 to 18 features**, recording the best model and feature set at each size — pushing peak accuracy higher than using all features at once.

### 🔗 Markov Chain Analysis

A stochastic model treating churn as a sequence of states, where the next state depends only on the current one.

<br>

## 📂 Dataset

The project uses the [Telco Customer Churn dataset](wa_fn-usec_-telco-customer-churn__1_.csv) — a public sample dataset describing a fictional telecom provider's customer base. It holds **7,043 customers across 21 attributes** (one row per customer), mixing categorical and numerical features.

| Feature            | Description                                                                        |
| :----------------- | :--------------------------------------------------------------------------------- |
| `customerID`       | Unique identifier for each customer                                                |
| `gender`           | Whether the customer is Male or Female                                             |
| `SeniorCitizen`    | Whether the customer is a senior citizen (1) or not (0)                            |
| `Partner`          | Whether the customer has a partner (Yes / No)                                      |
| `Dependents`       | Whether the customer has dependents (Yes / No)                                     |
| `tenure`           | Number of months the customer has stayed with the company                          |
| `PhoneService`     | Whether the customer has phone service (Yes / No)                                  |
| `MultipleLines`    | Whether the customer has multiple phone lines                                      |
| `InternetService`  | Customer's internet provider (DSL / Fiber optic / No)                              |
| `OnlineSecurity`   | Whether the customer has the online security add-on                                |
| `OnlineBackup`     | Whether the customer has the online backup add-on                                  |
| `DeviceProtection` | Whether the customer has the device protection add-on                              |
| `TechSupport`      | Whether the customer has the tech support add-on                                   |
| `StreamingTV`      | Whether the customer has streaming TV                                              |
| `StreamingMovies`  | Whether the customer has streaming movies                                          |
| `Contract`         | Contract term (Month-to-month / One year / Two year)                               |
| `PaperlessBilling` | Whether the customer uses paperless billing (Yes / No)                             |
| `PaymentMethod`    | How the customer pays (Electronic check, Mailed check, Bank transfer, Credit card) |
| `MonthlyCharges`   | Amount charged to the customer each month                                          |
| `TotalCharges`     | Total amount charged over the customer's tenure                                    |
| `Churn`            | **Target** — whether the customer left within the last month (Yes / No)            |

<br>

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HarshTanwar1/Churn_Analysis.git
cd Churn_Analysis
```

### 2. (Recommended) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 4. Launch the notebook

```bash
jupyter notebook Churn_Analysis.ipynb
```

> 💡 Keep `wa_fn-usec_-telco-customer-churn__1_.csv` in the same directory as the notebook — it's loaded via a relative path. Then run `Kernel → Restart & Run All` to reproduce the full analysis

<br>

## 📚 What I Learned

- The **CRISP-DM** workflow end-to-end — from framing a business question to interpreting model results.
- Cleaning and preparing a real-world, mixed-type dataset (type coercion, missing values, encoding).
- Translating statistics (variance, correlation, skewness) into **business insight**, not just numbers.
- Choosing the right visualization for categorical vs. numerical data.
- The strengths, weaknesses, and assumptions across **linear, distance-based, probabilistic, tree-based, and boosting** classifiers.
- **Hyperparameter tuning & cross-validation** with `GridSearchCV` and k-fold CV.
- That **feature selection matters** — an exhaustive combination search can beat using every variable at once.
- Modelling churn as a stochastic process with **Markov chains**.

<br>

## 🔭 Future Improvements

- ⚖️ **Address class imbalance** — apply class weighting or resampling to improve recall on the minority (churn) class.
- 📐 **Use richer metrics** — precision, recall, F1, and ROC-AUC are more meaningful than accuracy for imbalanced data.
- 🧱 **Modularize** — split data prep, EDA, and modelling into reusable `.py` files with a clear pipeline.
- 💾 **Persist trained models** with `joblib`/`pickle` instead of retraining each run.

<br>

---

<div align="center">

⭐ _If you found this project helpful or interesting, consider giving it a star!_ ⭐

</div>
