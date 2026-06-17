import numpy as np
import matplotlib.pyplot as plt
from Scripts.Preliminary_Predictions_S2 import *

# Compute raw residuals at hospital level
# Residual = actual demand - predicted demand
S2_test_results["residual_raw"] = S2_test_results["n"] - S2_test_results["predicted"]

# Aggregate data to daily level
# n         : total observed demand per day
# predicted : total predicted demand per day
# m         : number of hospital-level observations contributing to that day
daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    m=("residual_raw", "size"),
)

# Extract all hospital-level residuals
# These residuals form the empirical error distribution used in the bootstrap
res = S2_test_results["residual_raw"].values.astype(float)

# Create a reproducible random number generator
rng = np.random.default_rng(0)

def band_for_m(m, alpha=0.1, B=20000):
    """
    Generate a bootstrap-based prediction interval for a day
    containing m hospital-level observations.

    Parameters
    ----------
    m : int
        Number of hospital observations aggregated on the day.
    alpha : float
        Significance level (0.1 gives a 90% prediction interval).
    B : int
        Number of bootstrap simulations.

    Returns
    -------
    lo, hi : float
        Lower and upper bootstrap error quantiles.
    """

    # Draw m residuals with replacement for each bootstrap sample
    # and aggregate them to obtain a simulated daily error
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)

    # Extract lower and upper quantiles of the simulated error distribution
    lo, hi = np.quantile(sims, [alpha/2, 1-alpha/2])

    return lo, hi

# Compute bootstrap prediction interval errors for each day
daily["lo_err"], daily["hi_err"] = zip(*[band_for_m(int(m)) for m in daily["m"]])

# Convert error intervals into demand prediction intervals
daily["lower"] = daily["predicted"] + daily["lo_err"]
daily["upper"] = daily["predicted"] + daily["hi_err"]

# Display the resulting daily forecasts and prediction intervals
print(daily[["date", "n", "predicted", "lower", "upper"]].head())

