import pytest
import base64

VALID_CLAIM = {
    "claim_amount": 1000.0, "num_services": 3,
    "patient_age": 45, "provider_id": 7,
    "days_since_last_claim": 90
}

def test_missing_column_returns_422(client, mocker):
    mocker.patch('api.flask_app.requests.post')  # don't call BentoML
    # upload CSV missing 'provider_id'
    csv_missing_provider_id = (
        "claim_amount,num_services,patient_age,days_since_last_claim\n"
        "1000.0,3,45,90\n"
    )
    b64 = base64.b64encode(csv_missing_provider_id.encode("utf-8")).decode("utf-8")
    response = client.post(
        "/predict",
        data={"file": f"data:text/csv;base64,{b64}"},
    )
    assert response.status_code == 422

def test_no_file_data_returns_400(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400

def test_negative_claim_amount_rejected():
    from api.schemas import Claim
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Claim(**{**VALID_CLAIM, "claim_amount": -500})