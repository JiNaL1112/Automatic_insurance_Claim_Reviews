from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bentoml_url: str = "http://127.0.0.1:3000/predict"
    flask_port: int = 5005
    flask_debug: bool = False
    log_level: str = "INFO"

settings = Settings()