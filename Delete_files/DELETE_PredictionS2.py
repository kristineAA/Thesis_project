import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import cleaned data
from CleanDataPreliminary import *

S2 = dataset_filtered.copy()
# add day of week
S2['day_of_week'] = dataset_filtered['date'].dt.dayofweek
# add binary weekend column
S2['is_weekend'] = S2['day_of_week'].apply(lambda x: 1 if x >=5 else 0)
# add month column
S2['month'] = S2['date'].dt.month
# add year column
S2['year'] = S2['date'].dt.year
# add day of month column
S2['day_of_month'] = S2['date'].dt.day
# Rearrange columns
S2 = S2[['date', 'day_of_week', 'is_weekend', 'month', 'year', 'day_of_month', 'is_holiday', 'name', 'size', 'Hospital ID', 'y_lat', 'x_lon', 'region', 'n']]

# Make name and region categorical variables
S2['name'] = S2['name'].astype('category')
S2['region'] = S2['region'].astype('category')

X = S2.iloc[:, 1:13]
y = S2.iloc[:, 13].values

# cutoff = last 14 days
cutoff = S2['date'].max() - pd.Timedelta(days=14)

train_data_S2 = S2[S2['date'] < cutoff]
test_data_S2  = S2[S2['date'] >= cutoff]

# set seed for reproducibility
np.random.seed(43)
# predict next 14 days with XGBoost model
X_train_S2 = train_data_S2.iloc[:, 1:13]
y_train_S2 = train_data_S2.iloc[:, 13].values
xgb_train_S2 = xgb.DMatrix(X_train_S2, y_train_S2, enable_categorical=True)

n = 1200
params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.03,
    }
model = xgb.train(params=params, dtrain=xgb_train_S2, num_boost_round=n)

# evaluate on test set
X_test_S2 = test_data_S2.iloc[:, 1:13]
xgb_test_S2 = xgb.DMatrix(X_test_S2, enable_categorical=True)
predicted_values_S2 = model.predict(xgb_test_S2)
predicted_values_S2 = np.round(predicted_values_S2)

# Aggregate predictions by date
S2_test_results = test_data_S2.copy()
S2_test_results['predicted'] = predicted_values_S2

# Also aggregate actual demand by date
S2_daily_agg = S2_test_results.groupby('date').agg({
    'n': 'sum',
    'predicted': 'sum'
}).reset_index()


results_S2 = dict()
# Calculate evaluation metrics
mae_S2 = mean_absolute_error(test_data_S2['n'], predicted_values_S2)
mse_S2 = mean_squared_error(test_data_S2['n'], predicted_values_S2)
rmse_S2 = np.sqrt(mse_S2)
std_S2 = np.std(test_data_S2['n'] - predicted_values_S2)
results_S2 = {
    'MAE': mae_S2,
    'MSE': mse_S2,
    'RMSE': rmse_S2,
    'STD': std_S2
    }