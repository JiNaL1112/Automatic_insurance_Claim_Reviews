import base64
import io
import time
import uuid

import pandas as pd
import requests
from flask import Flask, g, render_template, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_flask_exporter import PrometheusMetrics
from pydantic import ValidationError

from api.config import settings
from api.logger import get_logger
from api.schemas import ClaimBatch

START_TIME = time.time()
logger = get_logger(__name__)

app = Flask(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
metrics = PrometheusMetrics(app)

# Static app info label exposed on every metric
metrics.info('flask_app_info', 'Flask fraud detection app', version='1.0')

# Business metrics
claims_processed_total = Counter(
    "claims_processed_total",
    "Total number of claims submitted for prediction",
)
anomalies_detected_total = Counter(
    "anomalies_detected_total",
    "Total number of claims flagged as anomalies",
)
normal_predictions_total = Counter(
    "normal_predictions_total",
    "Total number of claims classified as normal",
)
prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total prediction failures by error type",
    ["error_type"],   # labels: bentoml_unavailable | bentoml_error | validation_error
)

# Latency
bentoml_request_duration_seconds = Histogram(
    "bentoml_request_duration_seconds",
    "Time spent waiting for BentoML /predict response",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
csv_rows_per_request = Histogram(
    "csv_rows_per_request",
    "Number of claim rows per uploaded CSV",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

# Live anomaly rate (gauge — recomputed per batch)
anomaly_rate_gauge = Gauge(
    "anomaly_rate_last_batch",
    "Fraction of claims flagged as anomalies in the most recent batch (0–1)",
)

# ── Hard limits ───────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 5 * 1024 * 1024          # 5 MB
ALLOWED_MIME_PREFIX = "data:text/csv"        # only accept CSV data-URIs

BENTOML_URL = settings.bentoml_url
FEATURES = [
    "claim_amount",
    "num_services",
    "patient_age",
    "provider_id",
    "days_since_last_claim",
]


# ── Request lifecycle ─────────────────────────────────────────────────────────

@app.before_request
def assign_request_id() -> None:
    """Attach a unique request ID to every inbound request."""
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))


@app.after_request
def attach_request_id(response):
    """Echo the request ID back so clients/load-balancers can correlate logs."""
    response.headers["X-Request-ID"] = g.get("request_id", "-")
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    req_id = g.get("request_id", "-")

    # ── 1. Presence check ────────────────────────────────────────────────────
    file_data = request.form.get("file", "").strip()
    if not file_data:
        logger.warning("predict: no file data received [req=%s]", req_id)
        return {"error": "No file data received"}, 400

    # ── 2. Size guard ────────────────────────────────────────────────────────
    if len(file_data.encode("utf-8")) > MAX_UPLOAD_BYTES:
        logger.warning(
            "predict: upload too large (%d bytes) [req=%s]",
            len(file_data.encode("utf-8")),
            req_id,
        )
        return {"error": "File too large. Maximum allowed size is 5 MB."}, 413

    # ── 3. MIME type validation ──────────────────────────────────────────────
    if not file_data.startswith(ALLOWED_MIME_PREFIX):
        logger.warning(
            "predict: invalid MIME type in upload [req=%s] prefix=%s",
            req_id,
            file_data[:40],
        )
        return {"error": "Invalid file type. Only CSV files are accepted."}, 415

    # ── 4. Base64 decode ─────────────────────────────────────────────────────
    try:
        b64_payload = file_data.split(",", 1)[1]
        decoded_bytes = base64.b64decode(b64_payload)
    except Exception:
        logger.exception("predict: base64 decode failed [req=%s]", req_id)
        return {"error": "Could not decode the uploaded file."}, 400

    # ── 5. CSV parse + Pydantic validation ───────────────────────────────────
    try:
        df = pd.read_csv(io.StringIO(decoded_bytes.decode("utf-8")))
    except UnicodeDecodeError:
        return {"error": "File must be UTF-8 encoded."}, 400
    except Exception:
        logger.exception("predict: CSV parse failed [req=%s]", req_id)
        return {"error": "Could not parse the uploaded file as CSV."}, 400

    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        prediction_errors_total.labels(error_type="validation_error").inc()
        return {
            "error": "Missing required columns",
            "details": missing_cols,
        }, 422

    try:
        records = df[FEATURES].to_dict(orient="records")
        batch = ClaimBatch(data=records)
    except ValidationError as exc:
        prediction_errors_total.labels(error_type="validation_error").inc()
        return {"error": "Invalid data in CSV", "details": exc.errors()}, 422
    except Exception:
        logger.exception("predict: validation failed [req=%s]", req_id)
        prediction_errors_total.labels(error_type="validation_error").inc()
        return {"error": "Failed to validate claim data."}, 400

    # Record batch size
    csv_rows_per_request.observe(len(batch.data))

    # ── 6. Preserve claim_id if present ─────────────────────────────────────
    claim_ids = df["claim_id"] if "claim_id" in df.columns else None

    # ── 7. Forward to BentoML ────────────────────────────────────────────────
    try:
        with bentoml_request_duration_seconds.time():
            response = requests.post(
                BENTOML_URL,
                json={"data": [c.model_dump() for c in batch.data]},
                timeout=30,
                headers={"X-Request-ID": req_id},
            )
    except requests.exceptions.Timeout:
        logger.error("predict: BentoML timeout [req=%s]", req_id)
        prediction_errors_total.labels(error_type="bentoml_unavailable").inc()
        return {"error": "Prediction service timed out. Please try again."}, 503
    except requests.exceptions.RequestException:
        logger.exception("predict: BentoML unreachable [req=%s]", req_id)
        prediction_errors_total.labels(error_type="bentoml_unavailable").inc()
        return {"error": "Prediction service is temporarily unavailable."}, 503

    if response.status_code != 200:
        logger.error(
            "predict: BentoML returned %d [req=%s] body=%s",
            response.status_code,
            req_id,
            response.text[:200],
        )
        prediction_errors_total.labels(error_type="bentoml_error").inc()
        return {"error": "Prediction service returned an unexpected error."}, 500

    predictions = response.json()["predictions"]

    # ── 8. Update business metrics ───────────────────────────────────────────
    n_anomalies = predictions.count(-1)
    n_normal = predictions.count(1)
    batch_size = len(predictions)

    claims_processed_total.inc(batch_size)
    anomalies_detected_total.inc(n_anomalies)
    normal_predictions_total.inc(n_normal)

    # Anomaly rate for this batch (0.0 – 1.0)
    anomaly_rate = n_anomalies / batch_size if batch_size > 0 else 0.0
    anomaly_rate_gauge.set(anomaly_rate)

    logger.info(
        "predict: processed %d claims, %d anomalies (%.1f%%) [req=%s]",
        batch_size,
        n_anomalies,
        anomaly_rate * 100,
        req_id,
    )

    # ── 9. Build result ──────────────────────────────────────────────────────
    result_df = df[FEATURES].copy()
    if claim_ids is not None:
        result_df.insert(0, "claim_id", claim_ids.values)
    result_df["Prediction"] = predictions
    result_df["Status"] = result_df["Prediction"].map({1: "✅ Normal", -1: "🚨 Anomaly"})

    return render_template(
        "result.html",
        tables=[result_df.to_html(classes="data", header=True, index=False)],
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "service": "flask-app",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }, 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        debug=settings.flask_debug,
        port=settings.flask_port,
    )