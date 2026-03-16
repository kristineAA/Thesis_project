import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import cleaned data
from CleanDataPreliminary import *

S2 = dataset_filtered.copy()

# features
S2['day_of_week'] = S2['date'].dt.dayofweek
S2['is_weekend'] = (S2['day_of_week'] >= 5).astype(int)
S2['month'] = S2['date'].dt.month
S2['year'] = S2['date'].dt.year
S2['day_of_month'] = S2['date'].dt.day

S2 = S2[['date', 'day_of_week', 'is_weekend', 'month', 'year', 'day_of_month',
         'is_holiday', 'name', 'size', 'Hospital ID', 'y_lat', 'x_lon', 'region', 'n']]

S2['name'] = S2['name'].astype('category')
S2['region'] = S2['region'].astype('category')

# -------- time-based split: train / val / test (last 14 days test, previous 14 days val) --------
test_days = 14
val_days = 14

max_date = S2['date'].max()
cutoff_test = max_date - pd.Timedelta(days=test_days)
cutoff_val  = cutoff_test - pd.Timedelta(days=val_days)

train_data_S2 = S2[S2['date'] < cutoff_val]
val_data_S2   = S2[(S2['date'] >= cutoff_val) & (S2['date'] < cutoff_test)]
test_data_S2  = S2[S2['date'] >= cutoff_test]

X_train = train_data_S2.iloc[:, 1:13]
y_train = train_data_S2.iloc[:, 13].values
X_val   = val_data_S2.iloc[:, 1:13]
y_val   = val_data_S2.iloc[:, 13].values
X_test  = test_data_S2.iloc[:, 1:13]
y_test  = test_data_S2.iloc[:, 13].values

dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dval   = xgb.DMatrix(X_val,   y_val,   enable_categorical=True)
dtest  = xgb.DMatrix(X_test,  y_test,  enable_categorical=True)

# reproducibility
np.random.seed(43)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.03,
    # optional but often helpful:
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
    # 'min_child_weight': 1,
    # 'reg_lambda': 1.0,
}

num_boost_round = 1200
early_stopping_rounds = 50

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=100
)

# -------- evaluate on validation --------
val_pred = np.round(model.predict(dval))
val_mae = mean_absolute_error(y_val, val_pred)
val_mse = mean_squared_error(y_val, val_pred)
val_rmse = np.sqrt(val_mse)
val_std = np.std(y_val - val_pred)

results_val = {"MAE": val_mae, "MSE": val_mse, "RMSE": val_rmse, "STD": val_std}

# -------- evaluate on test --------
test_pred = np.round(model.predict(dtest))
test_mae = mean_absolute_error(y_test, test_pred)
test_mse = mean_squared_error(y_test, test_pred)
test_rmse = np.sqrt(test_mse)
test_std = np.std(y_test - test_pred)

results_test = {"MAE": test_mae, "MSE": test_mse, "RMSE": test_rmse, "STD": test_std}

# -------- daily aggregation on test --------
S2_test_results = test_data_S2.copy()
S2_test_results['predicted'] = test_pred

S2_daily_agg = S2_test_results.groupby('date').agg(
    n=('n', 'sum'),
    predicted=('predicted', 'sum')
).reset_index()
