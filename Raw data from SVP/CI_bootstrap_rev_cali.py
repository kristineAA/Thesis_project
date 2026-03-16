import numpy as np
import matplotlib.pyplot as plt
from Predictions_S2_cali import *

res = S2_calib_results["residual_raw"].values.astype(float)

daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    m=("predicted", "size"),
)
daily = daily[0:365]


rng = np.random.default_rng(0)

def band_and_probs_for_m(m, alpha=0.25, B=20000):
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)

    # bounds
    lo, hi = np.quantile(sims, [alpha/2, 1 - alpha/2])

    # empirical probabilities
    p_low = np.mean(sims <= lo)
    p_mid = np.mean((sims > lo) & (sims < hi))
    p_high = np.mean(sims >= hi)

    return lo, hi, p_low, p_mid, p_high


results = [band_and_probs_for_m(int(m)) for m in daily["m"]]

daily["lo_err"]   = [r[0] for r in results]
daily["hi_err"]   = [r[1] for r in results]
daily["p_low"]    = [r[2] for r in results]
daily["p_mid"]    = [r[3] for r in results]
daily["p_high"]   = [r[4] for r in results]

daily["lower"] = daily["predicted"] + daily["lo_err"]
daily["upper"] = daily["predicted"] + daily["hi_err"]

print(daily[[
    "date", "n", "predicted",
    "lower", "upper",
    "p_low", "p_mid", "p_high"
]].head())