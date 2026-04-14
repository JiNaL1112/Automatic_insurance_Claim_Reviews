#!/bin/bash
set -e

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