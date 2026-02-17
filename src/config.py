from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_customers: int = 200
    max_days: int = 180
    churn_threshold_days: int = 60
    test_size: float = 0.3
    random_state: int = 42
    model_path: str = "models/model.joblib"
