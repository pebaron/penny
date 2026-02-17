from datetime import datetime
from src.config import Config
from src.data import generate_transactions
from src.features import build_features, add_target
from src.model import train_eval_save
from src.viz import plot_recency, plot_recency_by_churn, plot_corr

def main():
    cfg = Config()
    today = datetime(2026, 2, 15)

    tx = generate_transactions(cfg.n_customers, cfg.seed, today, cfg.max_days)
    feats = build_features(tx, today)
    df = add_target(feats, cfg.churn_threshold_days)

    auc, coefs = train_eval_save(
        df,
        feature_cols=["frequency", "monetary"],
        target_col="churn",
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        model_path=cfg.model_path,
    )

    print("ROC-AUC:", round(auc, 3))
    print(coefs)

    plot_recency(df)
    plot_recency_by_churn(df)
    plot_corr(df)

if __name__ == "__main__":
    main()
