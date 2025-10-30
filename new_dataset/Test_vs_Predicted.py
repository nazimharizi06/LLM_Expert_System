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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initialize
#load
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
#%% Regressors

lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
dtr = DecisionTreeRegressor(max_depth=15,random_state=17,criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=1000, random_state=17, max_depth=15)
gbr = GradientBoostingRegressor(max_depth=4, random_state=17, n_estimators=100, learning_rate=0.1)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=25,learning_rate=1)

#%% Test and analysis

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
#print(X_test)

#RMSE Calculations
def rms_error(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# def Accuracy_score3(orig, pred):
#     orig = np.array(orig)
#     pred = np.array(pred)
    
#     count = 0
#     for i in range(len(orig)):
#         if(pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
#             count += 1
#     a_20 = count / len(orig)
#     return a_20

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    mask_pos = orig >= 0
    mask_neg = orig < 0
    
    count_pos = np.sum((0.8 * orig[mask_pos] <= pred[mask_pos]) & (pred[mask_pos] <= 1.2 * orig[mask_pos]))
    count_neg = np.sum((1.2 * orig[mask_neg] <= pred[mask_neg]) & (pred[mask_neg] <= 0.8 * orig[mask_neg]))
    
    a_20 = (count_pos + count_neg) / len(orig)
    return a_20

#Statistical calculations
#SHOULD BE TEST NOT TRAIN
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
rmse=rms_error(y_test,y_pred)
a_20 = Accuracy_score3(y_test,y_pred)

#Print statements
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2:",r2)
print("RSME:", rmse)
print("A_20:", a_20)

# print("Y_test\n",y_test)
# print("The number of values in y_test",len(y_test))
# print("Y_pred",y_pred)

#%% Export to excel Actual vs expected

#knn_X_test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR'])
#X_test_df = pd.DataFrame((X_test), columns=['PhaseNoise', 'SymbolRate', 'SNR'])


# Expected vs actual dataframe
results_df = pd.DataFrame({
    'Expected/Test': y_test,
    'Predicted': y_pred
})

file_path = "C:/Users/ryanj/Code/Research_LLM/excel/LLM.xlsx"

# Export
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    #X_test_df.to_excel(writer, sheet_name='ABR_Features', index=False, startrow=0, startcol=0)
    results_df.to_excel(writer, sheet_name='kNN_TvP', index=False, startrow=0, startcol=0)
# %%


