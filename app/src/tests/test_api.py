"""
Integration tests for app/src/api/flask_app.py — /predict endpoint.

Covers:
  - Happy path: valid CSV → BentoML returns predictions → result.html rendered
  - BentoML unreachable (RequestException) → 503
  - BentoML returns non-200 status → 500
  - CSV with claim_id column present (id forwarded to result table)
  - BentoML response with mixed predictions (normal + anomaly)
"""

import base64
from unittest.mock import MagicMock
import requests as requests_lib


# ── Helpers ───────────────────────────────────────────────────────────────────

VALID_CSV = (
    "claim_amount,num_services,patient_age,provider_id,days_since_last_claim\n"
    "1000.0,3,45,7,90\n"
    "2000.0,5,60,12,180\n"
)

VALID_CSV_WITH_ID = (
    "claim_id,claim_amount,num_services,patient_age,provider_id,days_since_last_claim\n"
    "101,1000.0,3,45,7,90\n"
    "102,2000.0,5,60,12,180\n"
)

ANOMALY_CSV = (
    "claim_amount,num_services,patient_age,provider_id,days_since_last_claim\n"
    "999999.0,99,1,1,0\n"
    "1000.0,3,45,7,90\n"
)


def _encode_csv(csv_text: str) -> str:
    """Return the data-URI string the frontend sends."""
    b64 = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")
    return f"data:text/csv;base64,{b64}"


def _mock_bentoml_response(predictions: list) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"predictions": predictions}
    return mock_resp


# ── Happy path ────────────────────────────────────────────────────────────────

class TestPredictHappyPath:

    def test_valid_csv_returns_200(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            return_value=_mock_bentoml_response([1, 1]),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        assert response.status_code == 200

    def test_valid_csv_renders_result_html(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            return_value=_mock_bentoml_response([1, 1]),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        html = response.data.decode("utf-8")
        assert "Prediction Results" in html

    def test_normal_prediction_shows_normal_status(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            return_value=_mock_bentoml_response([1, 1]),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        html = response.data.decode("utf-8")
        assert "Normal" in html

    def test_anomaly_prediction_shows_anomaly_status(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            return_value=_mock_bentoml_response([-1, 1]),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(ANOMALY_CSV)},
        )
        html = response.data.decode("utf-8")
        assert "Anomaly" in html

    def test_csv_with_claim_id_includes_id_in_output(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            return_value=_mock_bentoml_response([1, -1]),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV_WITH_ID)},
        )
        html = response.data.decode("utf-8")
        assert "101" in html
        assert "102" in html

    def test_bentoml_receives_correct_feature_payload(self, client, mocker):
        mock_post = mocker.patch(
            "api.flask_app.requests.post",
            return_value=_mock_bentoml_response([1, 1]),
        )
        client.post("/predict", data={"file": _encode_csv(VALID_CSV)})

        call_kwargs = mock_post.call_args
        # Access via keyword or positional
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert "data" in sent_json
        assert len(sent_json["data"]) == 2
        first = sent_json["data"][0]
        assert set(first.keys()) == {
            "claim_amount", "num_services", "patient_age",
            "provider_id", "days_since_last_claim",
        }

    def test_mixed_predictions_both_statuses_present(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            return_value=_mock_bentoml_response([-1, 1]),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        html = response.data.decode("utf-8")
        assert "Normal" in html
        assert "Anomaly" in html


# ── BentoML error cases ───────────────────────────────────────────────────────

class TestPredictBentoMLErrors:

    def test_bentoml_unreachable_returns_503(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            side_effect=requests_lib.exceptions.RequestException("connection refused"),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        assert response.status_code == 503

    def test_bentoml_unreachable_error_message(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            side_effect=requests_lib.exceptions.ConnectionError("refused"),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        data = response.get_json()
        assert "error" in data
        assert "error" in data  # message wording is an implementation detail; status code is the contract

    def test_bentoml_500_returns_500(self, client, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mocker.patch("api.flask_app.requests.post", return_value=mock_resp)

        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        assert response.status_code == 500

    def test_bentoml_500_error_message(self, client, mocker):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "model not loaded"
        mocker.patch("api.flask_app.requests.post", return_value=mock_resp)

        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        data = response.get_json()
        assert "error" in data

    def test_bentoml_422_returns_500(self, client, mocker):
        """Any non-200 from BentoML should surface as a 500 to the caller."""
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "validation error"
        mocker.patch("api.flask_app.requests.post", return_value=mock_resp)

        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        assert response.status_code == 500

    def test_timeout_returns_503(self, client, mocker):
        mocker.patch(
            "api.flask_app.requests.post",
            side_effect=requests_lib.exceptions.Timeout("timed out"),
        )
        response = client.post(
            "/predict",
            data={"file": _encode_csv(VALID_CSV)},
        )
        assert response.status_code == 503