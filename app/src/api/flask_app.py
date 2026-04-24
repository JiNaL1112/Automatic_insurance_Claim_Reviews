import base64
import io
import time
import uuid

import pandas as pd
import requests
from flask import Flask, g, render_template, request
from pydantic import ValidationError

from api.config import settings
from api.logger import get_logger
from api.schemas import ClaimBatch

START_TIME = time.time()
logger = get_logger(__name__)

app = Flask(__name__)

# ── Hard limits ───────────────────────────────────────────────────────────────
# 5 MB expressed in bytes.  base64 inflates ~33%, so a 5 MB encoded payload
# corresponds to ~3.75 MB of raw CSV — more than enough for any realistic batch.
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

    # ── 2. Size guard (checked on the raw string before any decoding) ────────
    if len(file_data.encode("utf-8")) > MAX_UPLOAD_BYTES:
        logger.warning(
            "predict: upload too large (%d bytes) [req=%s]",
            len(file_data.encode("utf-8")),
            req_id,
        )
        return {"error": "File too large. Maximum allowed size is 5 MB."}, 413

    # ── 3. MIME type validation ──────────────────────────────────────────────
    # Expected format: "data:text/csv;base64,<payload>"
    if not file_data.startswith(ALLOWED_MIME_PREFIX):
        logger.warning(
            "predict: invalid MIME type in upload [req=%s] prefix=%s",
            req_id,
            file_data[:40],
        )
        return {"error": "Invalid file type. Only CSV files are accepted."}, 415

    # ── 4. Base64 decode ─────────────────────────────────────────────────────
    try:
        # Everything after the first comma is the base64 payload
        b64_payload = file_data.split(",", 1)[1]
        decoded_bytes = base64.b64decode(b64_payload)
    except Exception:
        # Do NOT include the raw exception — it may leak internal detail
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

    # Missing required columns — tell the client *which* columns are missing,
    # not the raw KeyError which would expose internal variable names.
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        return {
            "error": "Missing required columns",
            "details": missing_cols,
        }, 422

    try:
        records = df[FEATURES].to_dict(orient="records")
        batch = ClaimBatch(data=records)
    except ValidationError as exc:
        # Pydantic errors are safe to surface — they reference field names only
        return {"error": "Invalid data in CSV", "details": exc.errors()}, 422
    except Exception:
        logger.exception("predict: validation failed [req=%s]", req_id)
        return {"error": "Failed to validate claim data."}, 400

    # ── 6. Preserve claim_id if present ─────────────────────────────────────
    claim_ids = df["claim_id"] if "claim_id" in df.columns else None

    # ── 7. Forward to BentoML ────────────────────────────────────────────────
    try:
        response = requests.post(
            BENTOML_URL,
            json={"data": [c.model_dump() for c in batch.data]},
            timeout=30,
            # Forward the request ID so BentoML logs can be correlated
            headers={"X-Request-ID": req_id},
        )
    except requests.exceptions.Timeout:
        logger.error("predict: BentoML timeout [req=%s]", req_id)
        return {"error": "Prediction service timed out. Please try again."}, 503
    except requests.exceptions.RequestException:
        logger.exception("predict: BentoML unreachable [req=%s]", req_id)
        return {"error": "Prediction service is temporarily unavailable."}, 503

    if response.status_code != 200:
        # Log the real error internally; return a generic message externally
        logger.error(
            "predict: BentoML returned %d [req=%s] body=%s",
            response.status_code,
            req_id,
            response.text[:200],    # cap log size
        )
        return {"error": "Prediction service returned an unexpected error."}, 500

    predictions = response.json()["predictions"]

    # ── 8. Build result ──────────────────────────────────────────────────────
    result_df = df[FEATURES].copy()
    if claim_ids is not None:
        result_df.insert(0, "claim_id", claim_ids.values)
    result_df["Prediction"] = predictions
    result_df["Status"] = result_df["Prediction"].map({1: "✅ Normal", -1: "🚨 Anomaly"})

    logger.info(
        "predict: processed %d claims, %d anomalies [req=%s]",
        len(result_df),
        (result_df["Prediction"] == -1).sum(),
        req_id,
    )

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