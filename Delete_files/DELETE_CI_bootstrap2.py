import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data 
from CleanDataPreliminary import * 
#from PredictionS2_tensorflow import * 
from Preliminary_Predictions_S2 import *

# residuals from validation (already computed)
val_residuals = y_val - np.round(final_model.predict(dval))

n_boot = 2000
alpha = 0.10   # 90% CI

boot_preds = np.zeros((n_boot, len(test_pred)))

rng = np.random.default_rng(42)

test_daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=('predicted','sum')
)

for b in range(n_boot):
    sampled_res = rng.choice(val_residuals, size=len(test_pred), replace=True)
    boot_preds[b, :] = test_pred + sampled_res

lower = np.quantile(boot_preds, alpha / 2, axis=0)
upper = np.quantile(boot_preds, 1 - alpha / 2, axis=0)

S2_test_results["ci_lower_90"] = np.round(lower)
S2_test_results["ci_upper_90"] = np.round(upper)

boot_daily = np.zeros((n_boot, len(S2_daily_agg)))

for b in range(n_boot):
    sampled_res = rng.choice(val_residuals, size=len(S2_test_results), replace=True)
    tmp = S2_test_results.copy()
    tmp["boot_pred"] = tmp["predicted"] + sampled_res

    daily = tmp.groupby("date")["boot_pred"].sum().values
    boot_daily[b, :] = daily

S2_daily_agg_boot = test_daily.copy()
S2_daily_agg_boot["ci_lower_90"] = np.quantile(boot_daily, alpha / 2, axis=0)
S2_daily_agg_boot["ci_upper_90"] = np.quantile(boot_daily, 1 - alpha / 2, axis=0)

print(S2_daily_agg_boot.head())

