from __future__ import annotations
import bentoml
import pandas as pd

FEATURES = ['claim_amount', 'num_services', 'patient_age', 'provider_id', 'days_since_last_claim']

@bentoml.service(name="health_insurance_anomaly_detection_service")
class AnomalyDetectionService:

    def __init__(self):
        self.model = bentoml.sklearn.load_model(
            bentoml.models.get("health_insurance_anomaly_detector:latest")
        )

    @bentoml.api
    def predict(self, data: list[dict]) -> dict:
        df = pd.DataFrame(data)
        # Only use the features the model was trained on
        df = df[FEATURES]
        predictions = self.model.predict(df)
        return {"predictions": predictions.tolist()}