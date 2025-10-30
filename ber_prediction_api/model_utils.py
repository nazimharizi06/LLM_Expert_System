import pandas as pd
import joblib

MODEL_PATH = "abr_model.pkl"

def load_model_from_file(path=MODEL_PATH):
    return joblib.load(path)

def predict_direct_ber(model, input_features: dict):
    required_keys = ['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR']
    if not all(k in input_features for k in required_keys):
        raise ValueError(f"Missing one or more required keys: {required_keys}")

    df = pd.DataFrame([[
        input_features['PhaseNoise'],
        input_features['PilotLength'],
        input_features['PilotSpacing'],
        input_features['SymbolRate'],
        input_features['SNR']
    ]], columns=required_keys)

    prediction = model.predict(df)[0]
    return 0 if prediction < 1e-9 else round(prediction, 8)