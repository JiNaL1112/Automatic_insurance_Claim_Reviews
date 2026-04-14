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
logger = get_logger(__name__)

app = Flask(__name__)

BENTOML_URL = settings.bentoml_url
FEATURES = ['claim_amount', 'num_services', 'patient_age', 'provider_id', 'days_since_last_claim']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # --- 1. Extract the base64 file from the form ---
    file_data = request.form.get('file', '')
    if not file_data:
        return {"error": "No file data received"}, 400

    # --- 2. Decode the Base64-encoded file content ---
    try:
        # format: "data:text/csv;base64,<payload>"
        decoded_file = base64.b64decode(file_data.split(',')[1])
    except (IndexError, Exception) as e:
        return {"error": f"Failed to decode file: {e}"}, 400

    # --- 3. Parse CSV and validate with Pydantic ---
    try:
        df = pd.read_csv(io.StringIO(decoded_file.decode('utf-8')))
        records = df[FEATURES].to_dict(orient='records')
        batch = ClaimBatch(data=records)   # validates here
    except ValidationError as e:
        return {"error": "Invalid input", "details": e.errors()}, 422
    except KeyError as e:
        return {"error": f"Missing required column: {e}"}, 422
    except Exception as e:
        return {"error": f"Failed to parse CSV: {e}"}, 400

    # --- 4. Keep claim_id aside if present ---
    claim_ids = df['claim_id'] if 'claim_id' in df.columns else None

    # --- 5. Send validated features to BentoML ---
    try:
        response = requests.post(
            BENTOML_URL,
            json={"data": [c.model_dump() for c in batch.data]},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        logger.error("BentoML unreachable: %s", e)
        return {"error": f"BentoML service unreachable: {e}"}, 503

    if response.status_code != 200:
        logger.error("BentoML error %s: %s", response.status_code, response.text)
        return {"error": f"BentoML error {response.status_code}: {response.text}"}, 500

    predictions = response.json()['predictions']

    # --- 6. Build result DataFrame ---
    result_df = df[FEATURES].copy()
    if claim_ids is not None:
        result_df.insert(0, 'claim_id', claim_ids.values)
    result_df['Prediction'] = predictions
    result_df['Status'] = result_df['Prediction'].map({1: '✅ Normal', -1: '🚨 Anomaly'})

    return render_template(
        'result.html',
        tables=[result_df.to_html(classes='data', header=True, index=False)]
    )


@app.route('/health')
def health():
    return {
        "status": "ok",
        "service": "flask-app",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }, 200


if __name__ == '__main__':
    app.run(debug=settings.flask_debug, port=settings.flask_port)