from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="APP_",
    )

    bentoml_url: str = "http://127.0.0.1:3000/predict"
    flask_port: int = 5005
    flask_debug: bool = False
    log_level: str = "INFO"

    # Maximum raw upload size in bytes (5 MB).
    # Flask enforces this via MAX_CONTENT_LENGTH before our own check fires,
    # giving us two independent layers of protection.
    max_upload_bytes: int = 5 * 1024 * 1024   # 5 MB

    # Comma-separated list of trusted hostnames for the Flask server.
    # In production set APP_ALLOWED_HOSTS=fraud-detection.yourdomain.com
    # Leave as "*" only for local development (flask_debug=True).
    allowed_hosts: str = "*"


settings = Settings()