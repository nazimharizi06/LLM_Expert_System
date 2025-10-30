import pandas as pd
import joblib

MODEL_PATH = "knn_model.pkl"  # path for KNN 

def load_model_from_file(path=MODEL_PATH):
    return joblib.load(path)

def predict_received_power(model, input_features: dict):
    required_keys = ['Tx_power_dBm', 'G_tx_db', 'G_rx_db', 'distance']
    if not all(k in input_features for k in required_keys):
        raise ValueError(f"Missing one or more required keys: {required_keys}")

    df = pd.DataFrame([[
        input_features['Tx_power_dBm'],
        input_features['G_tx_db'],
        input_features['G_rx_db'],
        input_features['distance']
    ]], columns=required_keys)

    prediction = model.predict(df)[0]
    return round(prediction, 4)  # Round output
