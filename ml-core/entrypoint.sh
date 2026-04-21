#!/bin/bash
set -e

# Wait for MLflow to be ready before attempting to train
echo "Waiting for MLflow at $MLFLOW_TRACKING_URI ..."
until curl -sf "${MLFLOW_TRACKING_URI}/health" > /dev/null 2>&1; do
  echo "  MLflow not ready yet, retrying in 5s..."
  sleep 5
done
echo "MLflow is ready."

echo "Checking for registered model..."
if ! bentoml models get health_insurance_anomaly_detector:latest &>/dev/null; then
    echo "Model not found. Running pipeline to train and register..."
    python src/data/generate.py
    cd /app && python src/models/pipeline.py
    echo "Model registered successfully."
else
    echo "Model already registered. Skipping training."
fi

echo "Starting BentoML server..."
exec bentoml serve src/serving/service.py --host 0.0.0.0 --port 3000