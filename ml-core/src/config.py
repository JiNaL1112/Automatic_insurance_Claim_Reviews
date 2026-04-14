from pydantic_settings import BaseSettings

class MLSettings(BaseSettings):
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "Health Insurance Claim Anomaly Detection"
    bentoml_model_name: str = "health_insurance_anomaly_detector"
    bentoml_port: int = 3000

    class Config:
        env_file = ".env"

ml_settings = MLSettings()