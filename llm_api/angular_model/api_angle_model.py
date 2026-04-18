import math 
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
    Tx_power_dBm: float = Field(..., description="Transmit power in dBm")
    G_tx_db: float = Field(..., description="Transmit antenna gain proxy")
    G_rx_db: float = Field(..., description="Receive antenna gain proxy")
    distance: float = Field(..., description="Distance in meters")
    angle_deg: float = Field(..., description="Single antenna angle in degrees")
    modulation_qam: int = Field(..., description="Modulation order, e.g. 2, 4, 16, or 64")

@app.post("/predict-angle-power")
def predict_angle_power(data: AngleModelInput):
    predicted_power_watts = predict_received_power(data)

    if predicted_power_watts > 0:
        predicted_power_dbm = 10 * math.log10(predicted_power_watts * 1000)
    else:
        predicted_power_dbm = None

    return {
        "predicted_received_power_watts": float(predicted_power_watts),
        "predicted_received_power_dBm": round(predicted_power_dbm, 2) if predicted_power_dbm is not None else None
    }
