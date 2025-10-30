import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initilize
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
feature_importances_list = []

#%% Regressors
"""
kNN - k_neighbors = 2, weights = 'distance'				
ABR - estimators = 25, lr = 1, max_depth = 15				
RFR - estimators = 1000, max_depth = 15			
GBR - estimators = 75, lr = 0.11, max_depth = 12				
"""

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
dtr = DecisionTreeRegressor(max_depth=15, random_state=17)
rfr = RandomForestRegressor(n_estimators=1000, random_state=17, max_depth=15)
gbr = GradientBoostingRegressor(max_depth=12, random_state=17, n_estimators=75, learning_rate=0.11)
abr = AdaBoostRegressor(estimator=dtr, random_state=17, n_estimators=5, learning_rate=0.01)


#%%FI

# Calculation
for state in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)
    rfr.fit(X_train, y_train)
    feature_importances_list.append(rfr.feature_importances_)

feature_importances_df = pd.DataFrame(feature_importances_list, columns=['Tx_power_dBm', 'G_tx_db', 'G_rx_db', 'distance']).transpose()

feature_importances_df.insert(0, 'Feature', feature_importances_df.index)

feature_importances_df.reset_index(drop=True, inplace=True)

#Export
file_path = "C:/Users/ryanj/Code/Research_LLM/excel/LLM.xlsx"
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    feature_importances_df.to_excel(writer, sheet_name='ABRFeatureImportances', index=False)
    
#%% Correlation Coefficent


# correlation_matrix = Book1.corr()


# print(correlation_matrix)


# # To export this correlation matrix to an Excel file
# file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"

# with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
#     correlation_matrix.to_excel(writer, sheet_name='Correlation', index=False, startrow=0, startcol=0)

