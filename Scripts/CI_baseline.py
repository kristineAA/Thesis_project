import numpy as np
import matplotlib.pyplot as plt

# data
from Scripts.Preliminary_Predictions_S2 import *

errors = y_val - val_pred
q_low, q_high = np.quantile(errors, [0.05, 0.95])

test_lower = np.maximum(test_pred + q_low, 0)
test_upper = test_pred + q_high

test_daily = test_data_S2.groupby("date", as_index=False).agg(
    n=("n", "sum")
)

S2_test_results = test_data_S2.copy()
S2_test_results['predicted'] = test_pred
S2_test_results['lower_90'] = test_lower
S2_test_results['upper_90'] = test_upper

S2_daily_agg_baseline = S2_test_results.groupby('date').agg(
    n=('n', 'sum'),
    predicted=('predicted', 'sum'),
    lower_90=('lower_90', 'sum'),
    upper_90=('upper_90', 'sum')
).reset_index()

print(S2_daily_agg_baseline.head())