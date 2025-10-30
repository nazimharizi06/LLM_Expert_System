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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initilize
# Load .mat file
mat_data = loadmat('path_loss_matrix.mat')

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
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Number of estimators for AdaBoost

#trees: 5,10,15,20,25,30,35,40,45,50,75,100,250,500,750,1000
#5,15,25,30,40,50,75,100,250,500,750,1000,1250,1500,1750,2000
#estimators: 5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000
#5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000,5000,6000,7500,8500,10000
#learning rate: 0.01,0.05,0.06,0.07,0.08,0.09,0.1,0.11,.12,.13,.14,.15,.25,.5,1,2
#depth: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

trees = [5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000]
l_rate = [0.1]

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
            count += 1
    a_20 = count / len(orig)
    return a_20

#%% Training
cv_folds = 15
num_depths = len(trees)
num_epi = len(l_rate)

rows = num_depths * num_epi
all_cv_scores = np.zeros((rows, cv_folds))
all_cv_scores2 = np.zeros((rows, cv_folds))
all_cv_scores3 = np.zeros((rows, cv_folds))
current_row = 0

# Loop over depths and trees to fill the array
for depth_idx, rate in enumerate(l_rate):
    for tree_idx, n_trees in enumerate(trees):
        abr = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=8,random_state=17, criterion='squared_error'),
                                n_estimators=n_trees, learning_rate=rate, random_state=17)
        
        # cv_scores = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
        # Accuracy_Values = cross_val_score(abr, X_train, y_train, cv=cv_scores, scoring=custom_Scoring)
        
        # print('Trial #:', current_row)
        # print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
        # print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
        
        # all_cv_scores[current_row, :] = Accuracy_Values
        
        # Custom scoring a_20 calculation
        custom_Scoring3 = make_scorer(Accuracy_score3, greater_is_better=True)

        # Running cross validation
        CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
        Accuracy_Values3 = cross_val_score(abr, X_train, y_train, cv=CV, scoring=custom_Scoring3)
        
        print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
        print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(), 4))
        
        all_cv_scores2[current_row, :] = Accuracy_Values3

        #MAE
        custom_Scoring5 = make_scorer(mean_absolute_error, greater_is_better=True)

        # Running cross validation
        Accuracy_Values5 = cross_val_score(abr, X_train, y_train, cv=CV, scoring=custom_Scoring5)
        
        print('\nMAE for 5-fold Cross Validation:\n', Accuracy_Values5)
        print('\nFinal Average Accuracy MAE index of the model:', round(Accuracy_Values5.mean(), 4))
        
        all_cv_scores3[current_row, :] = Accuracy_Values5
        
        current_row += 1

#%% Export

# # SMAPE
# df_metrics = pd.DataFrame(all_cv_scores, columns=[f'Fold {i+1}' for i in range(cv_folds)])
# a_20
df_metricsA20 = pd.DataFrame(all_cv_scores2, columns=[f'Fold {i+1}' for i in range(cv_folds)])
#MAE
df_metrics_MAE = pd.DataFrame(all_cv_scores3, columns=[f'Fold {i+1}' for i in range(cv_folds)])

file_path = "C:/Users/ryanj/Code/Research_LLM/excel/LLM.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    # df_metrics.to_excel(writer, sheet_name='ABR_SMAPE_depth15', index_label='Depth')
    df_metricsA20.to_excel(writer, sheet_name='ABR12_A20_est', index_label='Depth')
    df_metrics_MAE.to_excel(writer, sheet_name='ABR12_MAE_est', index_label='Depth')