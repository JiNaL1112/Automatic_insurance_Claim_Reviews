# 🏥 Health Insurance Claims Fraud Detection

An end-to-end MLOps pipeline for detecting fraudulent health insurance claims using an **Isolation Forest** model, served via **BentoML** and exposed through a **Flask** web application. Experiment tracking is handled by **MLflow**.

## 🗂 Versions
| Tag   | What it covers                                      | Medium article |
|-------|-----------------------------------------------------|----------------|
| v1.0  | Local pipeline: train → MLflow → BentoML → Flask   | [Read →](link) |
| v2.0  | Docker Compose, GitHub Actions CI/CD, pytest, DVC   | [Read →](link) |


---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Benefits](#benefits)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Step-by-Step Usage](#step-by-step-usage)
  - [1. Generate Synthetic Data](#1-generate-synthetic-data)
  - [2. Start MLflow Server](#2-start-mlflow-server)
  - [3. Train the Model](#3-train-the-model)
  - [4. Download the Model Artifact](#4-download-the-model-artifact)
  - [5. Register the Model in BentoML](#5-register-the-model-in-bentoml)
  - [6. Serve the Model with BentoML](#6-serve-the-model-with-bentoml)
  - [7. Test the BentoML Endpoint](#7-test-the-bentoml-endpoint)
  - [8. Run the Flask Web Application](#8-run-the-flask-web-application)
- [API Reference](#api-reference)
- [Model Details](#model-details)

---

## Project Overview

This project builds a complete fraud detection pipeline for health insurance claims. It ingests claim records, trains an unsupervised anomaly detection model, tracks experiments with MLflow, serves predictions via BentoML, and exposes an interactive web interface via Flask where users can upload a CSV of claims and receive fraud predictions.

---

## Benefits

| Benefit | Description |
|---|---|
| ⚡ **Faster Processing** | ML models automate claim validation and fraud detection, significantly reducing processing time |
| 🎯 **Improved Accuracy** | Analyzes vast amounts of data to identify patterns and anomalies humans might miss |
| 🔧 **Increased Efficiency** | Automation frees up human adjusters to focus on complex or high-value claims |
| 💰 **Reduced Costs** | Automation and improved accuracy lead to lower processing costs and fewer fraudulent claims paid out |
| 😊 **Enhanced Customer Satisfaction** | Faster processing and quicker payouts lead to happier customers |
| 📊 **Better Risk Assessment** | More accurately predicts the likelihood and severity of future claims, improving underwriting and pricing |

---

## Architecture

```
CSV Upload (Browser)
        │
        ▼
  Flask App :5005
  (flask_app.py)
        │
        │  POST /predict  {"data": [...]}
        ▼
  BentoML Service :3000
  (service.py)
        │
        │  model.predict(df)
        ▼
  Isolation Forest Model
  (registered via BentoML)
        │
        ▼
  Predictions returned
  {-1: Anomaly, 1: Normal}
        │
        ▼
  Results Table (result.html)


  MLflow UI :5000  ←── Experiment tracking during training
```

---

## Project Structure

```
Automatic_insurance_Claim_Reviews/
│
├── synthetic_health_claims.py   # Generates synthetic claims dataset (1000 normal + 50 anomalies)
├── synthetic_health_claims.csv  # Generated dataset
│
├── isolation_model.py           # Trains Isolation Forest model + logs to MLflow
├── donwload_model.py            # Downloads model artifact from MLflow to model.pkl
├── register_model.py            # Registers model.pkl into BentoML model store
├── service.py                   # BentoML service definition (serves predictions on :3000)
│
├── flask_app.py                 # Flask web app (CSV upload UI on :5005)
├── test_claim.py                # Script to test BentoML endpoint directly
│
├── requirements.txt             # Python dependencies
│
└── templates/
    ├── index.html               # Upload form
    ├── result.html              # Prediction results table
    └── visualize.html           # Pie chart visualization
```

---

## Prerequisites

- Python 3.10+
- `pip` or `pip3`
- A running MLflow server (started in step 2)

---

## Setup & Installation

```bash
# Clone or navigate to the project directory
cd Automatic_insurance_Claim_Reviews

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes:

```
mlflow
pandas
numpy
scikit-learn
bentoml
flask
requests
matplotlib
```

---

## Step-by-Step Usage

### 1. Generate Synthetic Data

Creates `synthetic_health_claims.csv` with 1,000 normal claims and 50 injected anomalies (high claim amounts, high service counts).

```bash
python synthetic_health_claims.py
```

**Features generated:**

| Column | Description |
|---|---|
| `claim_id` | Unique claim identifier |
| `claim_amount` | Dollar value of the claim (normal: ~$1,000; anomalies: ~$10,000) |
| `num_services` | Number of services billed (normal: 1–9; anomalies: 10–20) |
| `patient_age` | Patient age (18–90) |
| `provider_id` | Provider identifier (1–50) |
| `days_since_last_claim` | Days elapsed since prior claim (0–365) |

---

### 2. Start MLflow Server

Open a terminal and start the MLflow tracking server. **Keep this terminal running.**

```bash
mlflow ui
```

MLflow UI will be available at: **http://127.0.0.1:5000**

---

### 3. Train the Model

Trains an Isolation Forest model and logs parameters, metrics, and the model artifact to MLflow.

```bash
python isolation_model.py
```

**What gets logged to MLflow:**

- Parameters: `n_estimators=100`, `contamination=0.05`
- Metrics: `train_anomaly_percentage`, `test_anomaly_percentage`
- Artifact: trained `sklearn` model

After running, open the MLflow UI → **Experiments** → **Health Insurance Claim Anomaly Detection** to view the run and copy the artifact URI.

---

### 4. Download the Model Artifact

Open `donwload_model.py` and update the `artifact_uri` with the URI copied from the MLflow UI. The format looks like:

```
mlflow-artifacts:/<experiment_id>/<run_id>/artifacts/model
```

Then run:

```bash
python donwload_model.py
```

This saves the model as `model.pkl` in the project directory.

---

### 5. Register the Model in BentoML

```bash
python register_model.py
```

Verify it was registered:

```bash
bentoml models list
```

You should see `health_insurance_anomaly_detector` listed with a version tag.

---

### 6. Serve the Model with BentoML

Open a **new terminal**, activate the virtual environment, then run:

```bash
bentoml serve service.py --reload
```

The BentoML service will be available at: **http://127.0.0.1:3000**

The `--reload` flag automatically restarts the service when `service.py` changes.

---

### 7. Test the BentoML Endpoint

With the BentoML service running, in another terminal run:

```bash
python test_claim.py
```

Expected output:

```
Test Case 1: ✅ Normal  (raw: 1)
Test Case 2: ✅ Normal  (raw: 1)
Test Case 3: 🚨 Anomaly (raw: -1)
...
```

---

### 8. Run the Flask Web Application

Open a **new terminal**, activate the virtual environment, then run:

```bash
python flask_app.py
```

The web app will be available at: **http://127.0.0.1:5005**

**How to use:**
1. Open the browser at `http://127.0.0.1:5005`
2. Upload `synthetic_health_claims.csv` (or any CSV with the required columns)
3. Click **Analyse Claims**
4. View the results table — claims flagged as `-1` are potential fraud, `1` are normal

---

## API Reference

### BentoML Endpoint

**`POST http://127.0.0.1:3000/predict`**

Request body:
```json
{
  "data": [
    {
      "claim_amount": 15000,
      "num_services": 10,
      "patient_age": 50,
      "provider_id": 3,
      "days_since_last_claim": 300
    }
  ]
}
```

Response:
```json
{
  "predictions": [-1]
}
```

| Prediction | Meaning |
|---|---|
| `1` | ✅ Normal claim |
| `-1` | 🚨 Potential fraud / anomaly |

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Isolation Forest |
| Library | `scikit-learn` |
| `n_estimators` | 100 |
| `contamination` | 0.05 (5% expected anomaly rate) |
| Training split | 80% train / 20% test |
| Input features | `claim_amount`, `num_services`, `patient_age`, `provider_id`, `days_since_last_claim` |
| Output | `1` (normal) or `-1` (anomaly) |

The Isolation Forest algorithm detects anomalies by isolating observations in a random forest. Anomalous points — like unusually high claim amounts or large numbers of billed services — require fewer splits to isolate, giving them higher anomaly scores.

---

## Port Reference

| Service | Port | URL |
|---|---|---|
| MLflow UI | 5000 | http://127.0.0.1:5000 |
| BentoML Service | 3000 | http://127.0.0.1:3000 |
| Flask Web App | 5005 | http://127.0.0.1:5005 |
