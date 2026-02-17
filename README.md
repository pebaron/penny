# 📊 Retail Churn Prediction – End-to-End ML Mini Project

## Overview

This project demonstrates a complete machine learning workflow for customer churn prediction in a retail-like environment.

It includes:

- Synthetic transaction data generation
- Feature engineering (RFM-style)
- Logistic regression modeling
- ROC-AUC evaluation
- Data visualization
- Model persistence

The goal is to simulate how customer behavior data can be transformed into actionable predictive insights.

---

## 🏪 Business Context

In retail, identifying customers at risk of churn is essential for:

- Targeted retention campaigns
- Marketing budget optimization
- Revenue protection
- Customer lifetime value management

In this simplified setup:

- A customer is labeled as **churned** if they have not purchased in the last 60 days.
- A logistic regression model predicts churn probability using behavioral features.

---

## 🧠 Methodology

### 1️⃣ Data Simulation

Synthetic transaction data is generated for 200 customers.

For each customer:
- Number of transactions follows a Poisson distribution
- Transaction amounts follow a Gamma distribution
- Transaction dates are randomly distributed within 180 days

This simulates realistic retail purchase patterns.

---

### 2️⃣ Feature Engineering

Customer-level features are created using transaction aggregation:

- **frequency** → number of transactions  
- **monetary** → total spend  
- **recency** → days since last purchase  
- **churn** → binary target variable (recency > 60 days)

This reflects a simplified **RFM (Recency, Frequency, Monetary)** framework commonly used in retail analytics.

---

### 3️⃣ Model Training

Model:
- Logistic Regression (scikit-learn)

Training setup:
- Train/test split (70/30)
- Stratified sampling
- Maximum iterations increased for stability

Evaluation metric:
- **ROC-AUC**

