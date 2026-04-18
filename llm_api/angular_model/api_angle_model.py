from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .angle_model_utils import predict_received_power
import math
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "angle_dataset.csv")

df = pd.read_csv(DATA_PATH)

MIN_ANGLE = float(df["angle_deg"].min())
MAX_ANGLE = float(df["angle_deg"].max())
SUPPORTED_MODS = sorted(df["modulation_qam"].unique().tolist())
SUPPORTED_TX_SIZES = sorted(df["tx_antenna_mm"].unique().tolist())
SUPPORTED_RX_SIZES = sorted(df["rx_antenna_mm"].unique().tolist())
SUPPORTED_PAIRS = set(
    df[["tx_antenna_mm", "rx_antenna_mm"]]
    .drop_duplicates()
    .apply(tuple, axis=1)
    .tolist()
)

# Set these to the actual constants used during collection
SUPPORTED_TX_POWER_DBM = 10.0
SUPPORTED_DISTANCE_M = 1.0

app = FastAPI(
    title="THz Rx Angular Power Prediction API",
    description="Predict waveform-derived received power from angular THz measurements.",
    version="1.0.0",
    servers=[{"url": "https://angular.llmresearchapi.com"}]
)

class AngleModelInput(BaseModel):
    Tx_power_dBm: float = Field(..., description="Transmit power in dBm. Current model only supports the collected constant value.")
    distance: float = Field(..., description="Distance in meters. Current model only supports the collected constant value.")
    tx_antenna_mm: int = Field(..., description="Transmit antenna size in mm. Supported values come from the dataset.")
    rx_antenna_mm: int = Field(..., description="Receive antenna size in mm. Supported values come from the dataset.")
    angle_deg: float = Field(..., description="Single antenna angle in degrees.")
    modulation_qam: int = Field(..., description="Supported modulation order from the dataset.")

@app.post("/predict-angle-power")
def predict_angle_power(data: AngleModelInput):
    if data.Tx_power_dBm != SUPPORTED_TX_POWER_DBM:
        raise HTTPException(
            status_code=400,
            detail=f"This model currently supports only Tx_power_dBm = {SUPPORTED_TX_POWER_DBM}."
        )

    if data.distance != SUPPORTED_DISTANCE_M:
        raise HTTPException(
            status_code=400,
            detail=f"This model currently supports only distance = {SUPPORTED_DISTANCE_M} m."
        )

    if data.tx_antenna_mm not in SUPPORTED_TX_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported tx_antenna_mm. Supported values: {SUPPORTED_TX_SIZES}"
        )

    if data.rx_antenna_mm not in SUPPORTED_RX_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported rx_antenna_mm. Supported values: {SUPPORTED_RX_SIZES}"
        )

    if (data.tx_antenna_mm, data.rx_antenna_mm) not in SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported antenna pair. Supported pairs: {sorted(list(SUPPORTED_PAIRS))}"
        )

    if not (MIN_ANGLE <= data.angle_deg <= MAX_ANGLE):
        raise HTTPException(
            status_code=400,
            detail=f"angle_deg out of supported range. Supported range: {MIN_ANGLE} to {MAX_ANGLE}"
        )

    if data.modulation_qam not in SUPPORTED_MODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported modulation_qam. Supported values: {SUPPORTED_MODS}"
        )

    predicted_power_watts = predict_received_power(data)

    if predicted_power_watts > 0:
        predicted_power_dbm = 10 * math.log10(predicted_power_watts * 1000)
    else:
        predicted_power_dbm = None

    return {
        "predicted_received_power_watts": float(predicted_power_watts),
        "predicted_received_power_dBm": round(predicted_power_dbm, 2) if predicted_power_dbm is not None else None,
        "supported_angle_range_deg": [MIN_ANGLE, MAX_ANGLE],
        "supported_modulations": SUPPORTED_MODS,
        "supported_antenna_pairs_mm": sorted(list(SUPPORTED_PAIRS))
    }