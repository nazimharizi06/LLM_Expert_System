import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Load dataset
mat_data = loadmat('final_path_loss_matrix.mat')
pathloss_mat = mat_data['pathloss_mat']

# define columns
columns = ["Tx_power_dBm", "G_tx_db", "G_rx_db", "distance", 
           "Rx_power_dBm_NF", "Rx_power_dBm_NF_Intensity", "Rx_power_dBm_Friss"]

#create DataFrame
df = pd.DataFrame(pathloss_mat, columns=columns)

#features and label
X = df[["Tx_power_dBm", "G_tx_db", "G_rx_db", "distance"]]
y = df["Rx_power_dBm_NF"]

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn.fit(X, y)

#save
joblib.dump(knn, 'knn_model.pkl')

print("KNN model trained and saved to knn_model.pkl.")
