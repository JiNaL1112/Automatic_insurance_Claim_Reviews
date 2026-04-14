from pydantic import BaseModel, Field, field_validator

class Claim(BaseModel):
    claim_amount: float = Field(..., gt=0, lt=1_000_000)
    num_services: int = Field(..., ge=1, le=100)
    patient_age: int = Field(..., ge=0, le=130)
    provider_id: int = Field(..., ge=1)
    days_since_last_claim: int = Field(..., ge=0, le=3650)

class ClaimBatch(BaseModel):
    data: list[Claim]

    @field_validator("data")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Claim batch must contain at least one record")
        return v