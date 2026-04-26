from __future__ import annotations

import bentoml
import pandas as pd
from pydantic import BaseModel, Field
from prometheus_client import Counter  

FEATURES = [
    "claim_amount",
    "num_services",
    "patient_age",
    "provider_id",
    "days_since_last_claim",
]

prediction_counter = Counter(
    'fraud_predictions_total',
    'Total predictions made',
    ['result']
)


class Claim(BaseModel):
    """
    Pydantic schema for a single claim record at the BentoML boundary.

    IMPORTANT: these bounds must stay in sync with app/src/api/schemas.py.
    Any change here must be mirrored there (and vice-versa) to prevent the
    Flask layer accepting values the model layer will reject, or the model
    layer silently accepting out-of-range values.
    """

    claim_amount: float = Field(..., gt=0, lt=1_000_000)
    num_services: int = Field(..., ge=1, le=100)
    patient_age: int = Field(..., ge=0, le=130)
    provider_id: int = Field(..., ge=1, le=99_999)   # matches app schemas.py
    days_since_last_claim: int = Field(..., ge=0, le=3650)


@bentoml.service(
    name="health_insurance_anomaly_detection_service",
    metrics={"enabled": True},   # exposes /metrics on :3000
)
class AnomalyDetectionService:

    def __init__(self):
        self.model = bentoml.sklearn.load_model(
            bentoml.models.get("health_insurance_anomaly_detector:latest")
        )

    @bentoml.api
    def predict(self, data: list[dict]) -> dict:
        if not data:
            return {"predictions": []}

        if len(data) > 10_000:
            raise ValueError("Batch too large: maximum 10 000 claims per request")

        validated = [Claim(**record) for record in data]
        df = pd.DataFrame([c.model_dump() for c in validated])[FEATURES]
        predictions = self.model.predict(df)
 
        for p in predictions:
            label = 'anomaly' if p == -1 else 'normal'
            prediction_counter.labels(result=label).inc()

        return {"predictions": predictions.tolist()}

    @bentoml.api
    def healthz(self) -> dict:
        return {
            "status": "ok",
            "model": "health_insurance_anomaly_detector",
            "features": FEATURES,
        }