import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Scripts.CleanDataPreliminary import *
import os, time
import matplotlib.pyplot as plt

# data + features

# Copy cleaned dataset to avoid changing the original data
S2 = dataset_filtered.copy()

# Create date-related features
S2['day_of_week'] = S2['date'].dt.dayofweek
S2['is_weekend'] = (S2['day_of_week'] >= 5).astype(int)
S2['month'] = S2['date'].dt.month
S2['year'] = S2['date'].dt.year
S2['day_of_month'] = S2['date'].dt.day

# Keep only columns used for the model
S2 = S2[['date','day_of_week','is_weekend','month','year','day_of_month',
         'is_holiday','name','size','Hospital ID','y_lat','x_lon','region','n']]

# Convert categorical variables so XGBoost can handle them correctly
S2['name'] = S2['name'].astype('category')
S2['region'] = S2['region'].astype('category')

# split

# Define the number of days used for testing and validation
test_days = 14
val_days = 90

# Define chronological cutoffs for the validation and test periods
max_date = S2['date'].max()
cutoff_test = max_date - pd.Timedelta(days=test_days)
cutoff_val  = cutoff_test - pd.Timedelta(days=val_days)

# Split data into training, validation, and test sets
train_data_S2 = S2[S2['date'] < cutoff_val]
val_data_S2   = S2[(S2['date'] >= cutoff_val) & (S2['date'] < cutoff_test)]
test_data_S2  = S2[S2['date'] >= cutoff_test]

# Separate feature columns and target column
X_train = train_data_S2.iloc[:, 1:13]
y_train = train_data_S2.iloc[:, 13].values
X_val   = val_data_S2.iloc[:, 1:13]
y_val   = val_data_S2.iloc[:, 13].values
X_test  = test_data_S2.iloc[:, 1:13]
y_test  = test_data_S2.iloc[:, 13].values

# Convert data to XGBoost's optimized matrix format
dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dval   = xgb.DMatrix(X_val,   y_val,   enable_categorical=True)
dtest  = xgb.DMatrix(X_test,  y_test,  enable_categorical=True)

# grid search

# Hyperparameter combinations to test
param_grid = {
    "max_depth": [3,4,5,6,7],
    "learning_rate": [0.01,0.03,0.1,0.2,0.3],
    "n_estimators": [300,800,1500],
}

# Fixed model parameters
base_params = {
    "objective": "reg:squarederror",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "seed": 43,
}

# Store the best hyperparameter setting found during grid search
best = {"mse": float("inf"), "params": None, "best_iteration": None}

# Train and validate one model for each hyperparameter combination
for md in param_grid["max_depth"]:
    for lr in param_grid["learning_rate"]:
        for n_est in param_grid["n_estimators"]:
            params = dict(base_params, max_depth=md, learning_rate=lr)

            # Train model and stop early if validation performance stops improving
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=n_est,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            # Predict validation set and evaluate using MSE
            pred = model.predict(dval)
            mse = mean_squared_error(y_val, pred)

            # Update best model if this combination performs better
            if mse < best["mse"]:
                best.update(
                    mse=mse,
                    params={"max_depth": md, "learning_rate": lr, "n_estimators": n_est},
                    best_iteration=model.best_iteration,
                )

# Build final parameter set from the best grid search result
final_params = dict(
    base_params,
    max_depth=best["params"]["max_depth"],
    learning_rate=best["params"]["learning_rate"]
)

# Use the number of boosting rounds selected by early stopping
final_rounds = best["best_iteration"] + 1

# Train final model with the selected parameters
final_model = xgb.train(
    params=dict(final_params, eval_metric="rmse"),
    dtrain=dtrain,
    num_boost_round=final_rounds,
    evals=[(dtrain,"train"),(dval,"val")],
    verbose_eval=False,
)

# evaluate on validation

# Predict validation demand and round to whole units
val_pred = np.round(final_model.predict(dval))

# Calculate validation metrics
val_mae = mean_absolute_error(y_val, val_pred)
val_mse = mean_squared_error(y_val, val_pred)
val_rmse = np.sqrt(val_mse)
val_std = np.std(y_val - val_pred)

# Store validation metrics in a dictionary
results_val = {
    "MAE": val_mae,
    "MSE": val_mse,
    "RMSE": val_rmse,
    "STD": val_std,
}

# predictions

# Predict test demand and round to whole units
test_pred = np.round(final_model.predict(dtest))

# Add predictions and residuals to the test data
S2_test_results = test_data_S2.copy()
S2_test_results['predicted'] = test_pred
S2_test_results['residual'] = S2_test_results['n'] - S2_test_results['predicted']

# Aggregate hospital-level observations into daily total demand
S2_daily_agg = S2_test_results.groupby("date", as_index=False).agg(
    n=('n','sum'),
    predicted=('predicted','sum')
)

# Calculate daily prediction residuals
S2_daily_agg['residual'] = S2_daily_agg['n'] - S2_daily_agg['predicted']