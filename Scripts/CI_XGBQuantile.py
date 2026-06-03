import numpy as np
import matplotlib.pyplot as plt

# data
from Scripts.Preliminary_Predictions_S2 import *

def train_quantile_model(alpha):
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 43,
    }

    model_qt = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1200,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return model_qt

model_q05 = train_quantile_model(0.05)
model_q50 = train_quantile_model(0.50)
model_q95 = train_quantile_model(0.95)

q05_pred = np.round(model_q05.predict(dtest))
q50_pred = np.round(model_q50.predict(dtest))
q95_pred = np.round(model_q95.predict(dtest))

test_daily = test_data_S2.groupby("date", as_index=False).agg(
    n=("n", "sum")
)

S2_test_results = test_data_S2.copy()
S2_test_results["predicted"] = q50_pred
S2_test_results["lower_90"] = q05_pred
S2_test_results["upper_90"] = q95_pred

S2_daily_agg = S2_test_results.groupby("date").agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    lower_90=("lower_90", "sum"),
    upper_90=("upper_90", "sum"),
).reset_index()

print(S2_daily_agg.head())