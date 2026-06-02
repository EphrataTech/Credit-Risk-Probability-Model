from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Input features for a single customer's latest transaction."""

    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str = Field(
        ..., example="2024-01-15T08:30:00Z"
    )
    PricingStrategy: int
    FraudResult: int = Field(0, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionId": "T1000",
                "BatchId": "B100",
                "AccountId": "A100",
                "SubscriptionId": "S100",
                "CustomerId": "C100",
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProviderId": "P1",
                "ProductId": "PR1",
                "ProductCategory": "airtime",
                "ChannelId": "web",
                "Amount": 150.0,
                "Value": 150.0,
                "TransactionStartTime": "2024-01-15T08:30:00Z",
                "PricingStrategy": 2,
                "FraudResult": 0,
            }
        }
    }


class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    is_high_risk: bool
