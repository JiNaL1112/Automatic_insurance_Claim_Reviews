import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
import bentoml
import pickle
import logging, json, sys
import os


logging.basicConfig(
    stream=sys.stdout,
    format='{"level":"%(levelname)s","msg":"%(message)s","time":"%(asctime)s"}',
    level=logging.INFO,
)

log = logging.getLogger(__name__)



def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def run_pipeline():
    params = load_params()

    dp = params["data"]
    mp = params["model"]
    ml = params["mlflow"]
    sp = params["serving"]

    # --- Load data ---
    df = pd.read_csv(dp["output_path"])
    features = mp["features"]
    X_train, X_test = train_test_split(
        df[features],
        test_size=mp["test_size"],
        random_state=mp["random_seed"]
    )

    # --- MLflow tracking ---
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ml["tracking_uri"]))
    mlflow.set_experiment(ml["experiment_name"])

    with mlflow.start_run():
        # Train
        model = IsolationForest(
            n_estimators=mp["n_estimators"],
            contamination=mp["contamination"],
            random_state=mp["random_seed"]
        )
        model.fit(X_train)

        # Evaluate
        train_preds = model.predict(X_train)
        test_preds  = model.predict(X_test)
        train_anomaly_pct = (train_preds == -1).mean() * 100
        test_anomaly_pct  = (test_preds  == -1).mean() * 100

        # Log to MLflow
        mlflow.log_params({
            "n_estimators":  mp["n_estimators"],
            "contamination": mp["contamination"],
            "features":      ",".join(features),
        })
        mlflow.log_metrics({
            "train_anomaly_pct": train_anomaly_pct,
            "test_anomaly_pct":  test_anomaly_pct,
        })
        mlflow.sklearn.log_model(model, "model")

        log.info("Train anomaly %: %.2f", train_anomaly_pct)
        log.info("Test  anomaly %: %.2f", test_anomaly_pct)

        # Validate before registering
        if not (2.0 <= test_anomaly_pct <= 10.0):
            raise ValueError(
                f"Test anomaly % {test_anomaly_pct:.2f} outside expected 2–10% range. "
                "Check data or contamination param."
            )

        # Save model.pkl locally
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Register in BentoML
        saved = bentoml.sklearn.save_model(sp["bentoml_model_name"], model)
        print(f"Model registered in BentoML: {saved}")

if __name__ == "__main__":
    run_pipeline()