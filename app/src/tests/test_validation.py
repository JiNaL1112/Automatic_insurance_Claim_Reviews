import pytest

VALID_CLAIM = {
    "claim_amount": 1000.0, "num_services": 3,
    "patient_age": 45, "provider_id": 7,
    "days_since_last_claim": 90
}

def test_missing_column_returns_422(client, mocker):
    mocker.patch('api.flask_app.requests.post')  # don't call BentoML
    # upload CSV missing 'provider_id'
    ...
    assert response.status_code == 422

def test_negative_claim_amount_rejected():
    from api.schemas import Claim
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Claim(**{**VALID_CLAIM, "claim_amount": -500})