import numpy as np
import matplotlib.pyplot as plt

# Import prepared data, DMatrix objects, and XGBoost setup
from Scripts.Preliminary_Predictions_S2 import *

def train_quantile_model(alpha):
    """
    Train an XGBoost quantile regression model for a given quantile level.

    Parameters
    ----------
    alpha : float
        Quantile level to estimate, e.g. 0.05, 0.50, or 0.95.

    Returns
    -------
    model_qt : xgb.Booster
        Trained XGBoost quantile regression model.
    """

    # Define model parameters for quantile regression
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 43,
    }

     # Train the model using validation data for early stopping
    model_qt = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1200,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return model_qt

# Train three separate quantile models:
# 5th percentile, median, and 95th percentile
model_q05 = train_quantile_model(0.05)
model_q50 = train_quantile_model(0.50)
model_q95 = train_quantile_model(0.95)

# Predict lower bound, median forecast, and upper bound on the test set
q05_pred = np.round(model_q05.predict(dtest))
q50_pred = np.round(model_q50.predict(dtest))
q95_pred = np.round(model_q95.predict(dtest))

# Aggregate observed demand to daily level
test_daily = test_data_S2.groupby("date", as_index=False).agg(
    n=("n", "sum")
)

# Store quantile predictions together with the original test data
S2_test_results = test_data_S2.copy()
S2_test_results["predicted"] = q50_pred
S2_test_results["lower_90"] = q05_pred
S2_test_results["upper_90"] = q95_pred

# Aggregate hospital-level predictions to daily level
S2_daily_agg = S2_test_results.groupby("date").agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    lower_90=("lower_90", "sum"),
    upper_90=("upper_90", "sum"),
).reset_index()

# Display the first rows of the daily aggregated results
print(S2_daily_agg.head())