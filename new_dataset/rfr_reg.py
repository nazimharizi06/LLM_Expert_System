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

trees = [5,10,15,25,35,50,75,100,125,250,400,500,600,750,1000]

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    mask_pos = orig >= 0
    mask_neg = orig < 0
    
    count_pos = np.sum((0.8 * orig[mask_pos] <= pred[mask_pos]) & (pred[mask_pos] <= 1.2 * orig[mask_pos]))
    count_neg = np.sum((1.2 * orig[mask_neg] <= pred[mask_neg]) & (pred[mask_neg] <= 0.8 * orig[mask_neg]))
    
    a_20 = (count_pos + count_neg) / len(orig)
    return a_20


#def Accuracy_score(orig, pred):
   # orig = np.exp(orig)  # Convert original values back from log scale
    #pred = np.exp(pred)  # Convert predicted values back from log scale
    #MAPE = np.mean((np.abs(orig - pred)) / orig)
   # return MAPE
    
#%% Training
cv_folds = 15
num_depths = 15
num_trees = len(trees)

rows = num_depths * num_trees

all_cv_scores = np.zeros((rows, cv_folds))
all_cv_scores2 = np.zeros((rows, cv_folds))
all_cv_scores3 = np.zeros((rows, cv_folds))
current_row = 0

# Loop over depths and trees to fill the array
for depth_idx, depth in enumerate(range(1, num_depths + 1)):
    for tree_idx, n_trees in enumerate(trees):
        rfr = RandomForestRegressor(n_estimators=n_trees, random_state=17, max_depth=depth)
        
        # Custom scoring a_20 calculation
        custom_Scoring3 = make_scorer(Accuracy_score3, greater_is_better=True)

        # Running cross validation
        CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
        Accuracy_Values3 = cross_val_score(rfr, X_train, y_train, cv=CV, scoring=custom_Scoring3)
        
        print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
        print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(), 4))
        
        all_cv_scores2[current_row, :] = Accuracy_Values3

        #MAE
        custom_Scoring5 = make_scorer(mean_absolute_error, greater_is_better=True)

        # Running cross validation
        Accuracy_Values5 = cross_val_score(rfr, X_train, y_train, cv=CV, scoring=custom_Scoring5)
        
        print('\nMAE for 5-fold Cross Validation:\n', Accuracy_Values5)
        print('\nFinal Average Accuracy MAE index of the model:', round(Accuracy_Values5.mean(), 4))
        
        all_cv_scores3[current_row, :] = Accuracy_Values5
        
        current_row += 1
#%% Export

# a_20
df_metricsA20 = pd.DataFrame(all_cv_scores2, columns=[f'Fold {i+1}' for i in range(cv_folds)])
#MAE
df_metrics_MAE = pd.DataFrame(all_cv_scores3, columns=[f'Fold {i+1}' for i in range(cv_folds)])

file_path = "C:/Users/ryanj/Code/Research_LLM/excel/LLM.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metricsA20.to_excel(writer, sheet_name='RFR_A20', index_label='Depth')
    df_metrics_MAE.to_excel(writer, sheet_name='RFR_MAE', index_label='Depth')