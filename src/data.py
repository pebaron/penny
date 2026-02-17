import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_transactions(n_customers: int, seed: int, today: datetime, max_days: int) -> pd.DataFrame:
    np.random.seed(seed)

    transactions = []
    for cust in range(1, n_customers + 1):
        n_tx = np.random.poisson(5)
        for _ in range(n_tx):
            days_ago = np.random.randint(1, max_days + 1)
            transactions.append({
                "customer_id": cust,
                "transaction_date": today - timedelta(days=int(days_ago)),
                "amount": float(np.random.gamma(2, 300))
            })

    return pd.DataFrame(transactions)
