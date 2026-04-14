from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    bentoml_url: str = "http://127.0.0.1:3000/predict"
    flask_port: int = 5005
    flask_debug: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()