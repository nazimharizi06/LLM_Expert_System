import numpy as np
import pandas as pd
import time
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

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

trial = 0
a_20values = []
mae_values = []

#log base 10
#returning the minimum value of our BER ex. 53->47, 58->42...

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    mask_pos = orig >= 0
    mask_neg = orig < 0
    
    count_pos = np.sum((0.8 * orig[mask_pos] <= pred[mask_pos]) & (pred[mask_pos] <= 1.2 * orig[mask_pos]))
    count_neg = np.sum((1.2 * orig[mask_neg] <= pred[mask_neg]) & (pred[mask_neg] <= 0.8 * orig[mask_neg]))
    
    a_20 = (count_pos + count_neg) / len(orig)
    return a_20

#A20
custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)
#MAE
custom_Scoring5 = make_scorer(mean_absolute_error, greater_is_better=True)


#%% Test
for neighbor in range(1,16):
    knn = KNeighborsRegressor(n_neighbors=neighbor, weights='distance')

    #Running cross validation
    CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values3 = cross_val_score(knn,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring3)
    a_20values.append(Accuracy_Values3)
    print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))


    CV_MAE = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values5 = cross_val_score(knn,X_train ,y_train,\
                                       cv=CV_MAE,scoring=custom_Scoring5)
    mae_values.append(Accuracy_Values5)
    print('\nMAE for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy MAE of the model:', round(Accuracy_Values5.mean(),4))
    
# %% Exporting to Excel
# df_metrics = pd.DataFrame({
#    'Mean Squared Error': mse_list,
#    'Mean Absolute Error': mae_list,
#    'R^2 Score': r2_list,
#    'MAPE': mape_values,
#    'RMSE': rmse_list
# })

df_metrics_a20 = pd.DataFrame(a_20values, index=range(1, 16), columns=range(1, 16))   
df_metrics_mae = pd.DataFrame(mae_values, index=range(1, 16), columns=range(1, 16))
    
file_path = "C:/Users/ryanj/Code/Research_LLM/excel/LLM.xlsx"

#Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics_a20.to_excel(writer, sheet_name='KNN_A20', index=False, startrow=0, startcol=0)
    df_metrics_mae.to_excel(writer, sheet_name='KNN_MAE', index=False, startrow=0, startcol=0)

# %%

