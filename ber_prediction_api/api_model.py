from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import joblib
from model_utils import predict_direct_ber

# load trained KNN
MODEL_PATH = "abr_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"❌ Model file not found at {MODEL_PATH}. Run train_and_save_model.py first.")

# input structure
class BERInput(BaseModel):
    PhaseNoise: float
    PilotLength: int
    PilotSpacing: int
    SymbolRate: float
    SNR: float

# Init FastAPI app
NGROK_URL = "https://unbribing-uncontumaciously-dortha.ngrok-free.dev" #UPDATE THIS!!!

app = FastAPI(
    title="THz BER Prediction API",
    description="Predict BER using a pre-trained AdaBoostRegressor model",
    version="1.0.0",
    servers=[{"url": NGROK_URL}]
)

# POST endpoint to predict BER
@app.post("/predict-ber")
def predict_ber(input_data: BERInput):
    try:
        ber = predict_direct_ber(model, input_data.model_dump())
        return {
            "input": input_data.model_dump(),
            "predicted_ber": ber
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))