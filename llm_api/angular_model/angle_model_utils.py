import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "angle_model.pkl")

model = joblib.load(MODEL_PATH)

def predict_received_power(input_data):
    features = np.array([[
        input_data.tx_angle_deg,
        input_data.rx_angle_deg,
        input_data.tx_antenna,
        input_data.rx_antenna,
        input_data.modulation_qam
    ]])

    prediction = model.predict(features)
    return float(prediction[0])
