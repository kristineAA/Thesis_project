import numpy as np
import matplotlib.pyplot as plt

# data
from CleanDataPreliminary import *
#from PredictionS2_tensorflow import *
from Preliminary_Predictions_S2 import *
from CI_XGBQuantile import train_quantile_model

# Import trained quantile models
model_q05 = train_quantile_model(0.05)
model_q50 = train_quantile_model(0.50)
model_q95 = train_quantile_model(0.95)

# Raw quantile predictions on test
q05_pred = np.round(model_q05.predict(dtest))
q50_pred = np.round(model_q50.predict(dtest))
q95_pred = np.round(model_q95.predict(dtest))

alpha = 0.10  # 90% PI

#Aggregate true test demand (ground truth)
test_daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=('predicted','sum')
)

# Predict quantiles on validation
q05_val = model_q05.predict(dval)
q95_val = model_q95.predict(dval)

# Predict quantiles on test
q05_test = model_q05.predict(dtest)
q95_test = model_q95.predict(dtest)
# Conformal nonconformity scores
scores = np.maximum(q05_val - y_val, y_val - q95_val)
scores = np.maximum(scores, 0.0)  # optional safety

# Compute conformal correction
n = len(scores)
qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")

# Calibrated prediction interval (CQR)
lower_90 = q05_test - qhat
upper_90 = q95_test + qhat

S2_daily_agg_cqr = S2_test_results.copy()
S2_daily_agg_cqr["lower_90"] = lower_90
S2_daily_agg_cqr["upper_90"] = upper_90

S2_daily_agg_cqr = test_daily.groupby("date").agg(
    lower_90=("lower_90", "sum"),
    upper_90=("upper_90", "sum"),
).reset_index()

print(S2_daily_agg_cqr.head())