from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    """
    Single health insurance claim record.

    Bounds here are the *source of truth* for the whole system.
    The BentoML service.py Claim model must stay in sync with these —
    see ml-core/src/serving/service.py.
    """

    claim_amount: float = Field(
        ...,
        gt=0,
        lt=1_000_000,
        description="Dollar value of the claim (0, 1 000 000)",
    )
    num_services: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of services billed",
    )
    patient_age: int = Field(
        ...,
        ge=0,
        le=130,
        description="Patient age in years",
    )
    provider_id: int = Field(
        ...,
        ge=1,
        le=99_999,           # Added upper bound — was unbounded, allowing garbage values
        description="Provider identifier",
    )
    days_since_last_claim: int = Field(
        ...,
        ge=0,
        le=3650,
        description="Days elapsed since the previous claim (max 10 years)",
    )


class ClaimBatch(BaseModel):
    data: list[Claim]

    @field_validator("data")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Claim batch must contain at least one record")
        return v

    @field_validator("data")
    @classmethod
    def must_not_exceed_max_batch(cls, v):
        # Prevent memory exhaustion from huge batches
        if len(v) > 10_000:
            raise ValueError("Batch too large: maximum 10 000 claims per request")
        return v