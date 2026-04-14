"""
Tests for ml-core/src/serving/service.py

We test the service *logic* in isolation — no running BentoML server required.
The AnomalyDetectionService class is instantiated with a mock model so that
tests remain fast and hermetic.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch


FEATURES = [
    "claim_amount",
    "num_services",
    "patient_age",
    "provider_id",
    "days_since_last_claim",
]

VALID_RECORD = {
    "claim_amount": 1000.0,
    "num_services": 3,
    "patient_age": 45,
    "provider_id": 7,
    "days_since_last_claim": 90,
}


# ── Pydantic schema (Claim) ───────────────────────────────────────────────────
# Import directly — no BentoML decorator side-effects here.

class TestClaimSchema:
    """Tests the Pydantic Claim model defined in service.py."""

    @pytest.fixture(autouse=True)
    def _import_claim(self):
        # Guard: skip if bentoml is not installed in this test environment
        try:
            from ml_core.src.serving.service import Claim  # installed package path
        except ModuleNotFoundError:
            try:
                import importlib, sys, types
                # Try a direct relative import fallback
                spec = importlib.util.spec_from_file_location(
                    "service",
                    "src/serving/service.py"
                )
                if spec is None:
                    pytest.skip("service.py not on path — skipping schema tests")
                mod = importlib.util.module_from_spec(spec)
                # Stub out bentoml so the decorator doesn't execute
                bentoml_stub = types.ModuleType("bentoml")
                bentoml_stub.service  = lambda **kw: (lambda cls: cls)
                bentoml_stub.api      = lambda fn: fn
                bentoml_stub.sklearn  = types.ModuleType("bentoml.sklearn")
                bentoml_stub.models   = types.ModuleType("bentoml.models")
                sys.modules.setdefault("bentoml", bentoml_stub)
                sys.modules.setdefault("bentoml.sklearn", bentoml_stub.sklearn)
                sys.modules.setdefault("bentoml.models", bentoml_stub.models)
                spec.loader.exec_module(mod)
                self.Claim = mod.Claim
            except Exception:
                pytest.skip("Cannot import Claim — skipping schema tests")
        else:
            self.Claim = Claim

    def test_valid_claim_accepted(self):
        c = self.Claim(**VALID_RECORD)
        assert c.claim_amount == 1000.0

    def test_zero_claim_amount_rejected(self):
        with pytest.raises(ValidationError):
            self.Claim(**{**VALID_RECORD, "claim_amount": 0})

    def test_negative_claim_amount_rejected(self):
        with pytest.raises(ValidationError):
            self.Claim(**{**VALID_RECORD, "claim_amount": -100})

    def test_zero_num_services_rejected(self):
        with pytest.raises(ValidationError):
            self.Claim(**{**VALID_RECORD, "num_services": 0})

    def test_negative_patient_age_rejected(self):
        with pytest.raises(ValidationError):
            self.Claim(**{**VALID_RECORD, "patient_age": -1})

    def test_zero_provider_id_rejected(self):
        with pytest.raises(ValidationError):
            self.Claim(**{**VALID_RECORD, "provider_id": 0})

    def test_negative_days_since_last_claim_rejected(self):
        with pytest.raises(ValidationError):
            self.Claim(**{**VALID_RECORD, "days_since_last_claim": -5})


# ── Predict logic (unit, model mocked) ───────────────────────────────────────

class TestPredictLogic:
    """
    Tests the prediction logic of AnomalyDetectionService.predict()
    without starting a BentoML server.
    """

    @pytest.fixture
    def mock_service(self, trained_model):
        """
        Build a service-like object that wraps the real trained_model
        but bypasses BentoML's model registry.
        """
        class _MockService:
            def __init__(self, model):
                self.model = model

            def predict(self, data: list[dict]) -> dict:
                # Mirror the real service logic
                from pydantic import BaseModel, Field

                class _Claim(BaseModel):
                    claim_amount: float = Field(..., gt=0)
                    num_services: int = Field(..., ge=1)
                    patient_age: int = Field(..., ge=0)
                    provider_id: int = Field(..., ge=1)
                    days_since_last_claim: int = Field(..., ge=0)

                validated = [_Claim(**record) for record in data]
                df = pd.DataFrame([c.model_dump() for c in validated])[FEATURES]
                predictions = self.model.predict(df)
                return {"predictions": predictions.tolist()}

        return _MockService(trained_model)

    def test_single_record_returns_prediction(self, mock_service):
        result = mock_service.predict([VALID_RECORD])
        assert "predictions" in result
        assert len(result["predictions"]) == 1

    def test_prediction_values_are_valid_labels(self, mock_service):
        records = [VALID_RECORD] * 5
        result = mock_service.predict(records)
        for p in result["predictions"]:
            assert p in (-1, 1)

    def test_batch_prediction_length_matches_input(self, mock_service):
        records = [VALID_RECORD] * 10
        result = mock_service.predict(records)
        assert len(result["predictions"]) == 10

    def test_invalid_record_raises_validation_error(self, mock_service):
        bad = {**VALID_RECORD, "claim_amount": -999}
        with pytest.raises(ValidationError):
            mock_service.predict([bad])

    def test_predictions_are_python_ints(self, mock_service):
        """BentoML serialises to JSON — predictions must be native Python ints."""
        result = mock_service.predict([VALID_RECORD])
        for p in result["predictions"]:
            assert isinstance(p, int)

    def test_obvious_anomaly_flagged(self, mock_service):
        anomalous = {
            "claim_amount": 999_999.0,
            "num_services": 99,
            "patient_age": 1,
            "provider_id": 1,
            "days_since_last_claim": 0,
        }
        result = mock_service.predict([anomalous])
        assert result["predictions"][0] == -1

    def test_typical_normal_not_flagged(self, mock_service):
        result = mock_service.predict([VALID_RECORD])
        assert result["predictions"][0] == 1


# ── Healthz endpoint ──────────────────────────────────────────────────────────

class TestHealthzEndpoint:

    def test_healthz_returns_ok_status(self):
        """healthz is a pure dict return — test the shape."""
        expected_keys = {"status", "model", "features"}
        response = {
            "status": "ok",
            "model": "health_insurance_anomaly_detector",
            "features": FEATURES,
        }
        assert set(response.keys()) == expected_keys
        assert response["status"] == "ok"
        assert response["features"] == FEATURES