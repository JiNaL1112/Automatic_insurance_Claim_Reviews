"""
Standalone evaluation script — can be run after training or called from pipeline.py.
Loads model.pkl and the test split, prints metrics, and optionally logs to MLflow.
"""
import pickle
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format='{"level":"%(levelname)s","msg":"%(message)s","time":"%(asctime)s"}',
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model_path: str = "model.pkl", params_path: str = "params.yaml") -> dict:
    params = load_params(params_path)
    dp = params["data"]
    mp = params["model"]
    ml = params["mlflow"]

    df = pd.read_csv(dp["output_path"])
    features = mp["features"]

    _, X_test = train_test_split(
        df[features],
        test_size=mp["test_size"],
        random_state=mp["random_seed"],
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)          # 1 = normal, -1 = anomaly
    anomaly_pct = (y_pred == -1).mean() * 100
    normal_pct  = (y_pred ==  1).mean() * 100

    metrics = {
        "eval_anomaly_pct": round(anomaly_pct, 4),
        "eval_normal_pct":  round(normal_pct, 4),
        "eval_n_samples":   len(y_pred),
    }

    log.info("Evaluation results: %s", metrics)

    # Optionally log to MLflow active run (no-op if no run is active)
    try:
        mlflow.set_tracking_uri(ml["tracking_uri"])
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
    except Exception as exc:
        log.warning("MLflow logging skipped: %s", exc)

    return metrics


if __name__ == "__main__":
    results = evaluate()
    for k, v in results.items():
        print(f"  {k}: {v}")