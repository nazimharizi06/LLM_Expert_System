from fastapi import FastAPI
from pydantic import BaseModel, Field
from .angle_model_utils import predict_received_power

app = FastAPI(
    title="THz Angular Power Prediction API",
    description="Predict waveform-derived received power from angular THz measurements.",
    version="1.0.0",
    servers=[{"url": "https://angular.llmresearchapi.com"}]
)

class AngleModelInput(BaseModel):
    tx_angle_deg: float = Field(..., description="Transmit antenna angle in degrees")
    rx_angle_deg: float = Field(..., description="Receive antenna angle in degrees")
    tx_antenna: int = Field(..., description="Transmit antenna size, e.g. 26 or 46")
    rx_antenna: int = Field(..., description="Receive antenna size, e.g. 26 or 46")
    modulation_qam: int = Field(..., description="Modulation order, e.g. 2, 4, 16, or 64")

@app.post("/predict-angle-power")
def predict_angle_power(data: AngleModelInput):
    prediction = predict_received_power(data)
    return {
        "predicted_received_power": prediction
    }
