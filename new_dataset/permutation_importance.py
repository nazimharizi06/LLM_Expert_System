import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance

#%% Initilize
#load
mat_data = loadmat('final_path_loss_matrix.mat')

# Extract matrix data (assuming it's a 2D array)
pathloss_mat = mat_data['pathloss_mat']

# Column names as per MATLAB code
columns = ["Tx_power_dBm", "G_tx_db", "G_rx_db", "distance", 
           "Rx_power_dBm_NF", "Rx_power_dBm_NF_Intensity", "Rx_power_dBm_Friss"]

# Create DataFrame
df = pd.DataFrame(pathloss_mat, columns=columns)

#x and y split
X = df[["Tx_power_dBm", "G_tx_db", "G_rx_db", "distance"]]
#print("X:", X)
y = df[ "Rx_power_dBm_NF"]
#print("y:\n", y)

#Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# selector = SelectKBest(f_regression, k=4)
# X_new = selector.fit_transform(X, y)
# selected_features = df.columns[selector.get_support(indices=True)] 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=17)

# Normalize FOR kNN ONLY
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=17)
# knn.fit(X_train, y_train)
# knn_predictions = knn.predict(X_test)

# Initialize list for feature importances
permutation_importances_list = []

#%% Regressors
"""
kNN - k_neighbors = 1, weights = 'distance'				
ABR - estimators = 25, lr = 1, max_depth = 15				
RFR - estimators = 5, max_depth = 15			
GBR - estimators = 75, lr = 0.11, max_depth = 12				
"""

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
dtr = DecisionTreeRegressor(max_depth=15, random_state=17)
rfr = RandomForestRegressor(n_estimators=5, random_state=17, max_depth=15)
gbr = GradientBoostingRegressor(max_depth=12, random_state=17, n_estimators=75, learning_rate=0.11)
abr = AdaBoostRegressor(estimator=dtr, random_state=17, n_estimators=5, learning_rate=0.01)

#%%Permutation Importance

#PI Calculation
for state in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)
    knn.fit(X_train, y_train)
    result = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=state)
    #summing to 1
    normalized_importances = result.importances_mean / np.sum(result.importances_mean)
    permutation_importances_list.append(normalized_importances)

#DataFrame for feature importances
feature_importances_df = pd.DataFrame(permutation_importances_list, columns=['Tx_power_dBm', 'G_tx_db', 'G_rx_db', 'distance']).transpose()

feature_importances_df.insert(0, 'Feature', feature_importances_df.index)
feature_importances_df.reset_index(drop=True, inplace=True)

#Export
file_path = "C:/Users/ryanj/Code/Research_LLM/excel/LLM.xlsx"
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    feature_importances_df.to_excel(writer, sheet_name='kNNPermutationImportances', index=False)


