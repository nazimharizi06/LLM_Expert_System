import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "angle_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "angle_model.pkl")

# Load the combined angular dataset
df = pd.read_csv("angle_dataset.csv")

# Input features
X = df[[
    "tx_angle_deg",
    "rx_angle_deg",
    "tx_antenna",
    "rx_antenna",
    "modulation_qam"
]]

# Target
y = df["received_power"]

# Create and train model
knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn.fit(X, y)

# Save model
joblib.dump(knn, "angle_model.pkl")

print("Angle model trained and saved to angle_model.pkl.")

