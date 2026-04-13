from __future__ import annotations
import bentoml
import pandas as pd

@bentoml.service(name="health_insurance_anomaly_detection_service")
class AnomalyDetectionService:

    def __init__(self):
        self.model = bentoml.sklearn.load_model(
            bentoml.models.get("health_insurance_anomaly_detector:latest")
        )

    @bentoml.api
    def predict(self, data: list[dict]) -> dict:
        df = pd.DataFrame(data)
        predictions = self.model.predict(df)
        return {"predictions": predictions.tolist()}