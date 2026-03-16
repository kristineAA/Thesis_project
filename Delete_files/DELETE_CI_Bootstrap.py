import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data 
from CleanDataPreliminary import * 
#from PredictionS2_tensorflow import * 
from Preliminary_Predictions_S2 import *


# ---- train residuals (row-level) ----
#train_pred = final_model.predict(dtrain)   # use final_model if that’s what you trained
#train_residuals = y_train - train_pred
val_residuals = y_val - final_model.predict(dval)

res_df = val_data_S2[['date']].copy()
res_df['residual'] = val_residuals

# ---- aggregate to DAILY residual sums (matches your aggregation) ----
daily_res = res_df.groupby('date', as_index=False).agg(residual=('residual', 'sum'))
daily_res = daily_res.sort_values('date').reset_index(drop=True)
r = daily_res['residual'].values
T = len(r)

# ---- daily point prediction on test (aggregate predictions by date) ----
point_pred = final_model.predict(dtest)
S2_test_tmp = test_data_S2[['date', 'n']].copy()
S2_test_tmp['pred_row'] = point_pred

test_daily = S2_test_tmp.groupby('date', as_index=False).agg(
    n=('n', 'sum'),
    pred=('pred_row', 'sum'),
)
test_daily = test_daily.sort_values('date').reset_index(drop=True)
H = len(test_daily)

# ---- Moving Block Bootstrap on DAILY residuals ----
BLOCK_SIZE = 7
N_BOOT = 1000
rng = np.random.default_rng(43)

boot_daily_pred = np.empty((N_BOOT, H), dtype=float)

for b in range(N_BOOT):
    # sample starting indices for blocks
    starts = rng.integers(0, T - BLOCK_SIZE + 1, size=int(np.ceil(H / BLOCK_SIZE)))
    boot_r = np.concatenate([r[s:s+BLOCK_SIZE] for s in starts])[:H]
    boot_daily_pred[b, :] = test_daily['pred'].values + boot_r

lower_90 = np.percentile(boot_daily_pred, 5, axis=0)
upper_90 = np.percentile(boot_daily_pred, 95, axis=0)
median   = np.percentile(boot_daily_pred, 50, axis=0)

S2_daily_agg_boot = test_daily.copy()
S2_daily_agg_boot['predicted'] = median
S2_daily_agg_boot['lower_90'] = lower_90
S2_daily_agg_boot['upper_90'] = upper_90

print(S2_daily_agg_boot.head())
