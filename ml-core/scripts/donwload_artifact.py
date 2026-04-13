import mlflow
import shutil
import os

# Set the MLflow tracking URI to your server's address
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define the MLflow artifact URI
artifact_uri = "mlflow-artifacts:/1/862d8885dec1495a9c8a1653c70b7419/artifacts/model"

# Download the artifact
artifact_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
print(f"Downloaded to: {artifact_path}")

# Specify the local path to save the model
local_path = "model.pkl"

# Move the file to the desired location
model_file = os.path.join(artifact_path, "model.pkl")

if os.path.exists(model_file):
    if os.path.exists(local_path):
        os.remove(local_path)
    shutil.copy(model_file, local_path)
    print(f"Model saved to: {local_path}")
else:
    print(f"model.pkl not found in {artifact_path}")
    print(f"Files available: {os.listdir(artifact_path)}")