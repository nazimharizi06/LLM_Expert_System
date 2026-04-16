from fastapi import FastAPI
from pydantic import BaseModel, Field
from .angle_model_utils import predict_received_power

app = FastAPI(
    title="THz Rx Angular Power Prediction API",
    description="Predict waveform-derived received power from angular THz measurements.",
    version="1.0.0",
    servers=[{"url": "https://angular.llmresearchapi.com"}]
)

class AngleModelInput(BaseModel):
    angle_deg: float = Field(..., description="Single antenna angle in degrees")
    modulation_qam: int = Field(..., description="Modulation order, e.g. 2, 4, 16, or 64")

@app.post("/predict-angle-power")
def predict_angle_power(data: AngleModelInput):
    prediction = predict_received_power(data)
    return {
        "predicted_received_power": prediction
    }
