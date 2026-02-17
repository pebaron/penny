import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_eval_save(df: pd.DataFrame, feature_cols, target_col: str, test_size: float, random_state: int, model_path: str):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    # before joblib.dump(...)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)


    joblib.dump(model, model_path)

    coefs = pd.Series(model.coef_[0], index=feature_cols)
    return auc, coefs
