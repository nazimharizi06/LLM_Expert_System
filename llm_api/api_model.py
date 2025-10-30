from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from model_utils import predict_received_power

# load trained KNN
MODEL_PATH = "knn_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

# input structure
class RxPowerInput(BaseModel):
    Tx_power_dBm: float
    G_tx_db: float
    G_rx_db: float
    distance: float

# FastAPI app
CLOUDFLARE_URL = "https://api.llmresearchapi.com"  

app = FastAPI(
    title="THz Rx Power Prediction API",
    description="Predict Rx Power using a pre-trained KNN model",
    version="1.0.0",
    servers=[{"url": CLOUDFLARE_URL}]
)

# POST endpoint to predict Rx power
@app.post("/predict-rx-power")
def predict_rx_power(input_data: RxPowerInput):
    try:
        rx_power = predict_received_power(model, input_data.model_dump())
        return {
            "input": input_data.model_dump(),
            "predicted_rx_power_dBm": rx_power
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
