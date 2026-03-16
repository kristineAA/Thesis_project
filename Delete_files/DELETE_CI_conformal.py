import numpy as np
import matplotlib.pyplot as plt

# data
from CleanDataPreliminary import *
#from PredictionS2_tensorflow import *
from Preliminary_Predictions_S2 import *

alpha = 0.10
val_pred = model.predict(dval)
err = y_val - val_pred

n = len(err)
q_lo = np.quantile(-err, np.ceil((n + 1) * (1 - alpha/2)) / n, method="higher")  # for lower bound
q_hi = np.quantile( err, np.ceil((n + 1) * (1 - alpha/2)) / n, method="higher")  # for upper bound

test_pred = model.predict(dtest)
lower_90 = test_pred - q_lo
upper_90 = test_pred + q_hi

test_daily = test_data_S2.groupby("date", as_index=False).agg(
    n=("n", "sum")
)

S2_test_results = test_data_S2.copy()
S2_test_results["predicted"] = np.round(test_pred)
S2_test_results["lower_90"] = lower_90
S2_test_results["upper_90"] = upper_90

S2_daily_agg_conf = S2_test_results.groupby("date").agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    lower_90=("lower_90", "sum"),
    upper_90=("upper_90", "sum"),
).reset_index()

print(S2_daily_agg_conf.head())