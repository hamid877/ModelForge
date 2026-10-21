from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class ChurnPredictionRequest(BaseModel):
    """Customer features required for churn prediction."""

    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0)
    monthly_charges: float = Field(gt=0)
    contract_type: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    support_tickets: int = Field(ge=0)


class DriftBatchRequest(BaseModel):
    """Customer feature records used for batch drift detection."""

    records: list[ChurnPredictionRequest] = Field(
        min_length=1,
        max_length=10_000,
    )
