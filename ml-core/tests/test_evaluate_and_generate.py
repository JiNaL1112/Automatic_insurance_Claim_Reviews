"""
Additional ml-core tests covering:
  - evaluate.evaluate() called end-to-end (actual module, not inline logic)
  - generate.main() function (writes CSV to disk from params.yaml)
"""

import json
import os
import pickle
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import IsolationForest


FEATURES = [
    "claim_amount",
    "num_services",
    "patient_age",
    "provider_id",
    "days_since_last_claim",
]


# ── evaluate.py end-to-end ────────────────────────────────────────────────────

class TestEvaluateModule:
    """
    Calls the real evaluate() function from evaluate.py.
    MLflow calls are patched out so no server is required.
    """

    @pytest.fixture
    def evaluate_fn(self):
        """Import evaluate() — skip gracefully if module unavailable."""
        try:
            from models.evaluate import evaluate
            return evaluate
        except ModuleNotFoundError:
            pytest.skip("models.evaluate not importable in this environment")

    @pytest.fixture
    def setup_evaluate(self, trained_model, mixed_df, tmp_path, params):
        """Write model.pkl, CSV, and params.yaml to tmp_path."""
        model_path = str(tmp_path / "model.pkl")
        csv_path = str(tmp_path / "synthetic_health_claims.csv")
        params_path = str(tmp_path / "params.yaml")

        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)

        mixed_df.to_csv(csv_path, index=False)

        # Override output_path so evaluate() reads from tmp_path
        p = dict(params)
        p["data"] = dict(p["data"])
        p["data"]["output_path"] = csv_path
        with open(params_path, "w") as f:
            yaml.dump(p, f)

        return model_path, params_path

    def test_evaluate_returns_dict(self, evaluate_fn, setup_evaluate):
        model_path, params_path = setup_evaluate
        with patch("mlflow.set_tracking_uri"), patch("mlflow.active_run", return_value=None):
            result = evaluate_fn(model_path=model_path, params_path=params_path)
        assert isinstance(result, dict)

    def test_evaluate_returns_expected_keys(self, evaluate_fn, setup_evaluate):
        model_path, params_path = setup_evaluate
        with patch("mlflow.set_tracking_uri"), patch("mlflow.active_run", return_value=None):
            result = evaluate_fn(model_path=model_path, params_path=params_path)
        assert set(result.keys()) == {"eval_anomaly_pct", "eval_normal_pct", "eval_n_samples"}

    def test_evaluate_pcts_sum_to_100(self, evaluate_fn, setup_evaluate):
        model_path, params_path = setup_evaluate
        with patch("mlflow.set_tracking_uri"), patch("mlflow.active_run", return_value=None):
            result = evaluate_fn(model_path=model_path, params_path=params_path)
        total = result["eval_anomaly_pct"] + result["eval_normal_pct"]
        assert abs(total - 100.0) < 1e-4

    def test_evaluate_n_samples_positive(self, evaluate_fn, setup_evaluate):
        model_path, params_path = setup_evaluate
        with patch("mlflow.set_tracking_uri"), patch("mlflow.active_run", return_value=None):
            result = evaluate_fn(model_path=model_path, params_path=params_path)
        assert result["eval_n_samples"] > 0

    def test_evaluate_n_samples_matches_test_split(self, evaluate_fn, setup_evaluate, params):
        model_path, params_path = setup_evaluate
        with patch("mlflow.set_tracking_uri"), patch("mlflow.active_run", return_value=None):
            result = evaluate_fn(model_path=model_path, params_path=params_path)
        # mixed_df has 210 rows (200 normal + 10 anomaly); 20% test split = 42
        from sklearn.model_selection import train_test_split
        expected_test_size = int(210 * params["model"]["test_size"])
        # Allow ±1 for rounding
        assert abs(result["eval_n_samples"] - expected_test_size) <= 1

    def test_evaluate_anomaly_pct_non_negative(self, evaluate_fn, setup_evaluate):
        model_path, params_path = setup_evaluate
        with patch("mlflow.set_tracking_uri"), patch("mlflow.active_run", return_value=None):
            result = evaluate_fn(model_path=model_path, params_path=params_path)
        assert result["eval_anomaly_pct"] >= 0

    def test_evaluate_mlflow_logging_skipped_gracefully(self, evaluate_fn, setup_evaluate):
        """evaluate() should not raise even if MLflow log_metrics errors."""
        model_path, params_path = setup_evaluate
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.active_run", return_value=MagicMock()), \
             patch("mlflow.log_metrics", side_effect=Exception("mlflow down")):
            # Should complete without raising
            result = evaluate_fn(model_path=model_path, params_path=params_path)
        assert "eval_anomaly_pct" in result

    def test_evaluate_missing_model_raises(self, evaluate_fn, setup_evaluate):
        _, params_path = setup_evaluate
        with pytest.raises(FileNotFoundError):
            evaluate_fn(model_path="/nonexistent/model.pkl", params_path=params_path)


# ── generate.main() ───────────────────────────────────────────────────────────

class TestGenerateMain:
    """
    Tests generate.main() — the CLI entry point that reads params.yaml
    and writes the CSV to disk.
    """

    @pytest.fixture
    def generate_main(self):
        try:
            from data.generate import main
            return main
        except ModuleNotFoundError:
            pytest.skip("data.generate not importable in this environment")

    @pytest.fixture
    def params_yaml(self, tmp_path):
        """Write a minimal params.yaml to tmp_path and return path."""
        p = {
            "data": {
                "output_path": str(tmp_path / "out.csv"),
                "num_normal_samples": 50,
                "num_anomalies": 5,
                "random_seed": 0,
            }
        }
        path = tmp_path / "params.yaml"
        with open(path, "w") as f:
            yaml.dump(p, f)
        return path, p

    def test_main_creates_csv(self, generate_main, params_yaml, monkeypatch):
        params_path, p = params_yaml
        monkeypatch.chdir(params_path.parent)
        generate_main()
        out = Path(p["data"]["output_path"])
        assert out.exists(), "main() should create the output CSV"

    def test_main_csv_has_correct_row_count(self, generate_main, params_yaml, monkeypatch):
        params_path, p = params_yaml
        monkeypatch.chdir(params_path.parent)
        generate_main()
        df = pd.read_csv(p["data"]["output_path"])
        expected = p["data"]["num_normal_samples"] + p["data"]["num_anomalies"]
        assert len(df) == expected

    def test_main_csv_has_required_columns(self, generate_main, params_yaml, monkeypatch):
        params_path, p = params_yaml
        monkeypatch.chdir(params_path.parent)
        generate_main()
        df = pd.read_csv(p["data"]["output_path"])
        for col in FEATURES:
            assert col in df.columns, f"Missing column: {col}"

    def test_main_csv_no_nulls(self, generate_main, params_yaml, monkeypatch):
        params_path, p = params_yaml
        monkeypatch.chdir(params_path.parent)
        generate_main()
        df = pd.read_csv(p["data"]["output_path"])
        assert df[FEATURES].isnull().sum().sum() == 0

    def test_main_is_deterministic(self, generate_main, params_yaml, monkeypatch):
        """Running main() twice with the same seed yields identical CSVs."""
        params_path, p = params_yaml
        monkeypatch.chdir(params_path.parent)

        generate_main()
        df1 = pd.read_csv(p["data"]["output_path"])

        generate_main()
        df2 = pd.read_csv(p["data"]["output_path"])

        pd.testing.assert_frame_equal(df1, df2)

    def test_main_creates_parent_dirs(self, generate_main, tmp_path, monkeypatch):
        """output_path with nested dirs should be created automatically."""
        nested_out = str(tmp_path / "nested" / "dir" / "out.csv")
        p = {
            "data": {
                "output_path": nested_out,
                "num_normal_samples": 20,
                "num_anomalies": 2,
                "random_seed": 1,
            }
        }
        params_path = tmp_path / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(p, f)
        monkeypatch.chdir(tmp_path)
        generate_main()
        assert Path(nested_out).exists()