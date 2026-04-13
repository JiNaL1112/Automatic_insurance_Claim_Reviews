import bentoml
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

saved = bentoml.sklearn.save_model("health_insurance_anomaly_detector", model)
print(f"Model saved: {saved}")