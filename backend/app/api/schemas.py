from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    """Customer features required for churn prediction."""

    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0)
    monthly_charges: float = Field(gt=0)
    contract_type: str = Field(min_length=1)
    support_tickets: int = Field(ge=0)
