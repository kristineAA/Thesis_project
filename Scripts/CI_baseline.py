import numpy as np
import matplotlib.pyplot as plt

# Import validation and test predictions from the forecasting script
from Scripts.Preliminary_Predictions_S2 import *

# Calculate validation residuals (actual demand - predicted demand)
errors = y_val - val_pred

# Compute the 5th and 95th percentiles of the residual distribution
q_low, q_high = np.quantile(errors, [0.05, 0.95])

# Apply the residual quantiles to the test predictions
# Lower bound is truncated at zero to avoid negative demand forecasts
test_lower = np.maximum(test_pred + q_low, 0)
test_upper = test_pred + q_high

# Aggregate actual demand to daily level
test_daily = test_data_S2.groupby("date", as_index=False).agg(
    n=("n", "sum")
)

# Store prediction results together with prediction interval bounds
S2_test_results = test_data_S2.copy()
S2_test_results['predicted'] = test_pred
S2_test_results['lower_90'] = test_lower
S2_test_results['upper_90'] = test_upper

# Aggregate predictions and prediction intervals to daily level
# by summing hospital-level values for each date
S2_daily_agg_baseline = S2_test_results.groupby('date').agg(
    n=('n', 'sum'),
    predicted=('predicted', 'sum'),
    lower_90=('lower_90', 'sum'),
    upper_90=('upper_90', 'sum')
).reset_index()

# Display the first few rows of the aggregated results
print(S2_daily_agg_baseline.head())