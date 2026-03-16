import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import cleaned data
from CleanDataPreliminary import *


import os, time
import xgboost as xgb
from tensorboardX import SummaryWriter

class XGBoostTensorBoardPredQuantiles(xgb.callback.TrainingCallback):
    def __init__(self, log_dir: str, dtrain=None, dval=None, prefix: str = "", log_every: int = 10):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prefix = (prefix + "/") if prefix else ""
        self.dtrain = dtrain
        self.dval = dval
        self.log_every = log_every

    def _log_pred_dist(self, model, dmat, name: str, step: int):
        yhat = model.predict(dmat).astype(np.float32)

        # Distribution/Histogram tab
        self.writer.add_histogram(f"{self.prefix}{name}/yhat", yhat, step)

        # Quantile curves in Scalars tab
        for q in (0.05, 0.10, 0.50, 0.90, 0.95):
            self.writer.add_scalar(
                f"{self.prefix}{name}/yhat_q{int(q*100)}",
                float(np.quantile(yhat, q)),
                step
            )

        # (optional) mean/std curves
        self.writer.add_scalar(f"{self.prefix}{name}/yhat_mean", float(np.mean(yhat)), step)
        self.writer.add_scalar(f"{self.prefix}{name}/yhat_std",  float(np.std(yhat)),  step)

    def after_iteration(self, model, epoch: int, evals_log):
        # log XGBoost eval metrics (rmse, etc.)
        for data_name, metrics in evals_log.items():
            for metric_name, values in metrics.items():
                self.writer.add_scalar(f"{self.prefix}{data_name}/{metric_name}", values[-1], epoch)

        # log prediction distributions every N rounds
        if epoch % self.log_every == 0:
            if self.dtrain is not None:
                self._log_pred_dist(model, self.dtrain, "train", epoch)
            if self.dval is not None:
                self._log_pred_dist(model, self.dval, "val", epoch)

        return False

    def after_training(self, model):
        self.writer.close()
        return model
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

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

# --- grids inspired by the site (depth, lr, estimators) ---
param_grid = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.1, 0.2, 0.3],
    "n_estimators": [300, 800, 1500],
}

base_params = {
    "objective": "reg:squarederror",
    # helpful defaults (keep or remove):
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "seed": 43,
}

best = {"mse": float("inf"), "params": None, "best_iteration": None}

for md in param_grid["max_depth"]:
    for lr in param_grid["learning_rate"]:
        for n_est in param_grid["n_estimators"]:
            params = dict(base_params, max_depth=md, learning_rate=lr)

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=n_est,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            pred = model.predict(dval)
            mse = mean_squared_error(y_val, pred)

            if mse < best["mse"]:
                best.update(
                    mse=mse,
                    params={"max_depth": md, "learning_rate": lr, "n_estimators": n_est},
                    best_iteration=model.best_iteration,
                )

print("Best val MSE:", best["mse"])
print("Best params:", best["params"])
print("Best iteration (early stop):", best["best_iteration"])


final_params = dict(base_params,
                    max_depth=best["params"]["max_depth"],
                    learning_rate=best["params"]["learning_rate"])

# use best_iteration (+1 because iteration is 0-based)
final_rounds = (best["best_iteration"] + 1) if best["best_iteration"] is not None else best["params"]["n_estimators"]

params = dict(final_params, eval_metric="rmse")


run_dir = f"runs/xgb_{time.strftime('%Y%m%d-%H%M%S')}"
tb_cb = XGBoostTensorBoardPredQuantiles(log_dir=run_dir, dtrain=dtrain, dval=dval, log_every=10)

final_model = xgb.train(
    params=dict(final_params, eval_metric="rmse"),
    dtrain=dtrain,
    num_boost_round=final_rounds,
    evals=[(dtrain, "train"), (dval, "val")],
    callbacks=[tb_cb, xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    verbose_eval=False,
)
print("TensorBoard logdir:", run_dir)
test_pred = np.round(final_model.predict(dtest))

# -------- evaluate on validation --------
val_pred = np.round(final_model.predict(dval))
val_mae = mean_absolute_error(y_val, val_pred)
val_mse = mean_squared_error(y_val, val_pred)
val_rmse = np.sqrt(val_mse)
val_std = np.std(y_val - val_pred)

results_val = {"MAE": val_mae, "MSE": val_mse, "RMSE": val_rmse, "STD": val_std}

# -------- evaluate on test --------
test_pred = np.round(final_model.predict(dtest))
test_mae = mean_absolute_error(y_test, test_pred)
test_mse = mean_squared_error(y_test, test_pred)
test_rmse = np.sqrt(test_mse)
test_std = np.std(y_test - test_pred)

results_test = {"MAE": test_mae, "MSE": test_mse, "RMSE": test_rmse, "STD": test_std}

# -------- daily aggregation on test --------
S2_test_results = test_data_S2.copy()
S2_test_results['predicted'] = test_pred

S2_daily_agg = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum")
)

S2_daily_agg["residual"] = S2_daily_agg["n"] - S2_daily_agg["predicted"]

S2_test_results["residual"] = (S2_test_results["n"] - S2_test_results["predicted"]).astype(np.float32)

with SummaryWriter(log_dir=run_dir) as w:
    w.add_histogram("test_raw/residuals", S2_test_results["residual"].values, 0)
    w.add_histogram("test_daily_agg/residuals", S2_daily_agg["residual"].values, 0)




