import pytest
import pandas as pd
import numpy as np
import pickle
import tempfile
import os
import yaml
from sklearn.ensemble import IsolationForest


# ── Minimal params.yaml structure mirroring ml-core/params.yaml ──────────────

@pytest.fixture
def params():
    return {
        "data": {
            "output_path": "synthetic_health_claims.csv",
            "num_normal_samples": 200,
            "num_anomalies": 10,
            "random_seed": 42,
        },
        "model": {
            "n_estimators": 10,        # small for speed in tests
            "contamination": 0.05,
            "random_seed": 42,
            "test_size": 0.2,
            "features": [
                "claim_amount",
                "num_services",
                "patient_age",
                "provider_id",
                "days_since_last_claim",
            ],
        },
        "mlflow": {
            "tracking_uri": "http://127.0.0.1:5000",
            "experiment_name": "Test Experiment",
        },
        "serving": {
            "bentoml_model_name": "health_insurance_anomaly_detector",
            "port": 3000,
        },
        "app": {
            "port": 5005,
            "bentoml_url": "http://127.0.0.1:3000/predict",
        },
    }


FEATURES = [
    "claim_amount",
    "num_services",
    "patient_age",
    "provider_id",
    "days_since_last_claim",
]


@pytest.fixture
def params_file(params, tmp_path):
    """Write params dict to a temporary params.yaml and return its path."""
    path = tmp_path / "params.yaml"
    with open(path, "w") as f:
        yaml.dump(params, f)
    return str(path)


@pytest.fixture
def normal_df():
    """200 rows of realistic normal claim data."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "claim_amount":          rng.uniform(100, 5_000, n),
        "num_services":          rng.integers(1, 10, n),
        "patient_age":           rng.integers(18, 80, n),
        "provider_id":           rng.integers(1, 50, n),
        "days_since_last_claim": rng.integers(1, 365, n),
    })


@pytest.fixture
def anomaly_df():
    """10 rows of obviously anomalous claims."""
    rng = np.random.default_rng(99)
    n = 10
    return pd.DataFrame({
        "claim_amount":          rng.uniform(500_000, 999_999, n),
        "num_services":          rng.integers(90, 100, n),
        "patient_age":           rng.integers(0, 5, n),
        "provider_id":           rng.integers(1, 5, n),
        "days_since_last_claim": rng.integers(0, 1, n),
    })


@pytest.fixture
def mixed_df(normal_df, anomaly_df):
    """Combined dataset with known normals and anomalies."""
    return pd.concat([normal_df, anomaly_df], ignore_index=True)


@pytest.fixture
def trained_model(normal_df):
    """A fast-trained IsolationForest fixture."""
    model = IsolationForest(n_estimators=10, contamination=0.05, random_state=42)
    model.fit(normal_df[FEATURES])
    return model


@pytest.fixture
def model_pkl(trained_model, tmp_path):
    """Persist the trained model to a temp file; return its path."""
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(trained_model, f)
    return str(path)


@pytest.fixture
def csv_file(mixed_df, tmp_path):
    """Write the mixed dataset to a temp CSV; return its path."""
    path = tmp_path / "synthetic_health_claims.csv"
    mixed_df.to_csv(path, index=False)
    return str(path)