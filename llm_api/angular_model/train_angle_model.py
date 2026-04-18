import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "angle_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "angle_model.pkl")

df = pd.read_csv(DATA_PATH)

print("Dataset preview:")
print(df.head())
print("\nColumns:", df.columns.tolist())

X = df[[
    "tx_antenna_mm",
    "rx_antenna_mm",
    "angle_deg",
    "modulation_qam"
]]

y = df["received_power"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=3, weights="distance"))
])

model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print(f"Angle model trained and saved to {MODEL_PATH}.")