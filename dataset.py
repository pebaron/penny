import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_customers = 200

customers = pd.DataFrame({
    "customer_id": range(1, n_customers + 1),
    "registration_days_ago": np.random.randint(30, 365, n_customers)
})

transactions = []

today = datetime(2026, 2, 15)

for cust in customers["customer_id"]:
    n_tx = np.random.poisson(5)
    for _ in range(n_tx):
        days_ago = np.random.randint(1, 180)
        transactions.append({
            "customer_id": cust,
            "transaction_date": today - timedelta(days=int(days_ago)),
            "amount": np.random.gamma(2, 300)
        })

transactions = pd.DataFrame(transactions)

reference_date = today

features = transactions.groupby("customer_id").agg(
    frequency=("amount", "count"),
    monetary=("amount", "sum"),
    last_purchase=("transaction_date", "max")
).reset_index()

features["recency"] = (reference_date - features["last_purchase"]).dt.days

features = features.drop(columns=["last_purchase"])

features["churn"] = (features["recency"] > 60).astype(int)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = features[["frequency", "monetary"]]
y = features["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", round(auc, 3))

coef = pd.Series(model.coef_[0], index=X.columns)
print(coef)

import joblib
joblib.dump(model, "model.joblib")

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(features["recency"], bins=30)
plt.title("Distribution of Recency")
plt.xlabel("Days since last purchase")
plt.show()

sns.histplot(data=features, x="recency", hue="churn", bins=30, kde=True)
plt.title("Recency by Churn")
plt.show()

corr = features[["recency", "frequency", "monetary", "churn"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
