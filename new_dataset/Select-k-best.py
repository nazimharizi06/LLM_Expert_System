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
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

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

# Drop constant columns
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df = X_scaled_df.loc[:, X_scaled_df.var() > 1e-8]  # tiny threshold to be safe

# Now apply SelectKBest
k = min(1, X_scaled_df.shape[1])  # make sure k is not bigger than number of columns

selector = SelectKBest(k=k)
X_new = selector.fit_transform(X_scaled_df, y)
# print(f"Selected features for k={k}: {selected_features[::-1].tolist()}")

# Since X_new is a numpy array without column names, for the train-test split, you should use it directly.
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=17)


def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    mask_pos = orig >= 0
    mask_neg = orig < 0
    
    count_pos = np.sum((0.8 * orig[mask_pos] <= pred[mask_pos]) & (pred[mask_pos] <= 1.2 * orig[mask_pos]))
    count_neg = np.sum((1.2 * orig[mask_neg] <= pred[mask_neg]) & (pred[mask_neg] <= 0.8 * orig[mask_neg]))
    
    a_20 = (count_pos + count_neg) / len(orig)
    return a_20

custom_Scoring = make_scorer(Accuracy_score3, greater_is_better=True)
#%% Regressors

lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
dtr = DecisionTreeRegressor(max_depth=15,random_state=17, criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=5, random_state=17, max_depth=15)
gbr = GradientBoostingRegressor(max_depth=15, random_state=17, n_estimators=100, learning_rate=0.1)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=25,learning_rate=1)


#%% A-20
#custom scoring a_20 calulation
custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

#Running cross validation
CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
Accuracy_Values3 = cross_val_score(abr,X_train,y_train,\
                                   cv=CV,scoring=custom_Scoring3)

print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))

selected_features_mask = selector.get_support()
selected_features = X_scaled_df.columns[selected_features_mask]

print(f"Selected features for k={k}: {selected_features[::-1].tolist()}")
#%% Export

file_path = "C:/Users/ryanj/Code/Research_LLM/excel/LLM.xlsx"

#accuracy df
df_metrics = pd.DataFrame(Accuracy_Values3, index=range(1, len(Accuracy_Values3) + 1), columns=['Accuracy'])

#descending order
df_selected_features = pd.DataFrame(selected_features.tolist()[::-1], columns=['Selected Features'])

#Export
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='kNNkBestAcc1', index=False, startrow=0, startcol=0)
    df_selected_features.to_excel(writer, sheet_name='kNNkBestSF1', index=False, startrow=0, startcol=0)

