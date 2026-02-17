import pandas as pd

def build_features(transactions: pd.DataFrame, reference_date) -> pd.DataFrame:
    features = transactions.groupby("customer_id").agg(
        frequency=("amount", "count"),
        monetary=("amount", "sum"),
        last_purchase=("transaction_date", "max")
    ).reset_index()

    features["recency"] = (reference_date - features["last_purchase"]).dt.days
    return features.drop(columns=["last_purchase"])

def add_target(features: pd.DataFrame, churn_threshold_days: int) -> pd.DataFrame:
    df = features.copy()
    df["churn"] = (df["recency"] > churn_threshold_days).astype(int)
    return df
