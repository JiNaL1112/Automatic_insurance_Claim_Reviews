from __future__ import annotations
import bentoml
import pandas as pd
from pydantic import BaseModel, Field

FEATURES = ['claim_amount', 'num_services', 'patient_age', 'provider_id', 'days_since_last_claim']


class Claim(BaseModel):
    claim_amount: float = Field(..., gt=0)
    num_services: int = Field(..., ge=1)
    patient_age: int = Field(..., ge=0)
    provider_id: int = Field(..., ge=1)
    days_since_last_claim: int = Field(..., ge=0)


@bentoml.service(name="health_insurance_anomaly_detection_service")
class AnomalyDetectionService:

    def __init__(self):
        self.model = bentoml.sklearn.load_model(
            bentoml.models.get("health_insurance_anomaly_detector:latest")
        )

    @bentoml.api
    def predict(self, data: list[dict]) -> dict:
        # Validate each record through Pydantic then build DataFrame
        validated = [Claim(**record) for record in data]
        df = pd.DataFrame([c.model_dump() for c in validated])[FEATURES]
        predictions = self.model.predict(df)
        return {"predictions": predictions.tolist()}

    @bentoml.api
    def healthz(self) -> dict:
        return {
            "status": "ok",
            "model": "health_insurance_anomaly_detector",
            "features": FEATURES,
        }