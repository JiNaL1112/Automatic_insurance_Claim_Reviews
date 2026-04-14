from flask import Flask, render_template, request
import pandas as pd
import requests
import base64
import io
from config import settings
from schemas import Claim, ClaimBatch
from pydantic import ValidationError
from logger import get_logger
import time


START_TIME = time.time()


app = Flask(__name__)

BENTOML_URL = settings.bentoml_url
FEATURES = ['claim_amount', 'num_services', 'patient_age', 'provider_id', 'days_since_last_claim']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        df = pd.read_csv(io.StringIO(decoded_file.decode('utf-8')))
        records = df[FEATURES].to_dict(orient='records')
        batch = ClaimBatch(data=records)   # validates here
    except ValidationError as e:
        return {"error": "Invalid input", "details": e.errors()}, 422
    except KeyError as e:
        return {"error": f"Missing required column: {e}"}, 422

    # Decode the Base64 encoded file content
    decoded_file = base64.b64decode(file_data.split(',')[1])

    # Read into DataFrame
    df = pd.read_csv(io.StringIO(decoded_file.decode('utf-8')))

    # Keep claim_id aside if present
    claim_ids = df['claim_id'] if 'claim_id' in df.columns else None

    # Only send the model features to BentoML
    df_features = df[FEATURES]

    # Send to BentoML — wrap in {"data": [...]} as required
    response = requests.post(
        BENTOML_URL,
        json={"data": df_features.to_dict(orient='records')}
    )

    if response.status_code != 200:
        return f"BentoML error {response.status_code}: {response.text}", 500

    predictions = response.json()['predictions']

    # Build result DataFrame
    result_df = df_features.copy()
    if claim_ids is not None:
        result_df.insert(0, 'claim_id', claim_ids.values)
    result_df['Prediction'] = predictions
    result_df['Status'] = result_df['Prediction'].map({1: '✅ Normal', -1: '🚨 Anomaly'})

    return render_template('result.html', tables=[result_df.to_html(classes='data', header=True, index=False)])

if __name__ == '__main__':
    app.run(debug=True, port=5005)

@app.route('/health')
def health():
    return {
        "status": "ok",
        "service": "flask-app",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }, 200