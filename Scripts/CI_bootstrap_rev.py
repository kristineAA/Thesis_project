import numpy as np
import matplotlib.pyplot as plt
from Scripts.Preliminary_Predictions_S2 import *

# raw residuals (row-level) on test
S2_test_results["residual_raw"] = S2_test_results["n"] - S2_test_results["predicted"]

# daily sum + count
daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    m=("residual_raw", "size"),
)

res = S2_test_results["residual_raw"].values.astype(float)
rng = np.random.default_rng(0)

def band_for_m(m, alpha=0.1, B=20000):
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)
    lo, hi = np.quantile(sims, [alpha/2, 1-alpha/2])
    return lo, hi

daily["lo_err"], daily["hi_err"] = zip(*[band_for_m(int(m)) for m in daily["m"]])
daily["lower"] = daily["predicted"] + daily["lo_err"]
daily["upper"] = daily["predicted"] + daily["hi_err"]

print(daily[["date", "n", "predicted", "lower", "upper"]].head())

