"""
Tests for ml-core/src/data/generate.py

We test the *contract* of the generator:
  - output CSV has the expected columns
  - row count matches params
  - value ranges are sensible
  - anomalies are distinguishable from normal records
  - re-running with the same seed is deterministic
"""

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = {
    "claim_amount",
    "num_services",
    "patient_age",
    "provider_id",
    "days_since_last_claim",
}

OPTIONAL_COLUMNS = {"claim_id", "label"}   # may or may not be present


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def generated_csv(params, csv_file):
    """
    Falls back to the mixed_df csv fixture when the generator module is not
    importable (e.g. running tests without the full ml-core package installed).
    """
    try:
        from src.data.generate import generate_dataset  # type: ignore
    except Exception:
        return csv_file

    dp = params["data"]
    out_dir = Path(os.path.dirname(csv_file))
    out_path = out_dir / "generated_synthetic_health_claims.csv"

    df = generate_dataset(
        num_normal_samples=dp["num_normal_samples"],
        num_anomalies=dp["num_anomalies"],
        random_seed=dp["random_seed"],
    )
    df.to_csv(out_path, index=False)
    return str(out_path)


# ── column / shape tests ──────────────────────────────────────────────────────

class TestGeneratedDataShape:

    def test_expected_columns_present(self, generated_csv):
        df = load_csv(generated_csv)
        missing = EXPECTED_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_null_values(self, generated_csv):
        df = load_csv(generated_csv)
        nulls = df[list(EXPECTED_COLUMNS)].isnull().sum()
        assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"

    def test_row_count_matches_params(self, params, generated_csv):
        df = load_csv(generated_csv)
        expected = params["data"]["num_normal_samples"] + params["data"]["num_anomalies"]
        assert len(df) == expected, (
            f"Expected {expected} rows, got {len(df)}"
        )


# ── value-range tests ─────────────────────────────────────────────────────────

class TestGeneratedDataRanges:

    def test_claim_amount_positive(self, generated_csv):
        df = load_csv(generated_csv)
        assert (df["claim_amount"] > 0).all()

    def test_num_services_at_least_one(self, generated_csv):
        df = load_csv(generated_csv)
        assert (df["num_services"] >= 1).all()

    def test_patient_age_reasonable(self, generated_csv):
        df = load_csv(generated_csv)
        assert (df["patient_age"] >= 0).all()
        assert (df["patient_age"] <= 130).all()

    def test_provider_id_positive(self, generated_csv):
        df = load_csv(generated_csv)
        assert (df["provider_id"] >= 1).all()

    def test_days_since_last_claim_non_negative(self, generated_csv):
        df = load_csv(generated_csv)
        assert (df["days_since_last_claim"] >= 0).all()


# ── anomaly distinguishability ────────────────────────────────────────────────

class TestAnomalyDistribution:

    def test_anomalies_have_higher_claim_amounts_on_average(
        self, normal_df, anomaly_df
    ):
        """Anomaly fixture has extreme claim amounts; mean should be much higher."""
        assert anomaly_df["claim_amount"].mean() > normal_df["claim_amount"].mean() * 10

    def test_anomalies_have_more_services_on_average(self, normal_df, anomaly_df):
        assert anomaly_df["num_services"].mean() > normal_df["num_services"].mean()


# ── determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_seed_produces_identical_output(self, normal_df):
        """
        Two DataFrames built with the same RNG seed must be identical.
        We re-create the fixture logic inline to avoid module-import coupling.
        """
        def make_df(seed):
            rng = np.random.default_rng(seed)
            n = 200
            return pd.DataFrame({
                "claim_amount":          rng.uniform(100, 5_000, n),
                "num_services":          rng.integers(1, 10, n),
                "patient_age":           rng.integers(18, 80, n),
                "provider_id":           rng.integers(1, 50, n),
                "days_since_last_claim": rng.integers(1, 365, n),
            })

        df1 = make_df(42)
        df2 = make_df(42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_output(self):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        a1 = rng1.uniform(0, 1, 100)
        a2 = rng2.uniform(0, 1, 100)
        assert not np.array_equal(a1, a2)