import numpy as np
import matplotlib.pyplot as plt
from Predictions_S2_CV import *

res = S2_calib_results["residual_raw"].values.astype(float)

daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    m=("predicted", "size"),
)

rng = np.random.default_rng(0)

#percentiles = [10, 25, 50, 75, 90]
percentiles = [5, 20, 35, 50, 65, 80, 95]

def scenario_stats_for_m(m, percentiles, B=20000):
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)
    qvals = np.quantile(sims, np.array(percentiles) / 100.0)
    probs = []
    for q in qvals:
        probs.append(np.mean(sims <= q))   # will be approx the percentile level
    return qvals, probs

results = [scenario_stats_for_m(int(m), percentiles) for m in daily["m"]]

for i, p in enumerate(percentiles):
    daily[f"err_p{p}"] = [r[0][i] for r in results]
    daily[f"p{p}"] = daily["predicted"] + daily[f"err_p{p}"]
    daily[f"prob_p{p}"] = [r[1][i] for r in results]

output_cols = (
    ["date", "n", "predicted"]
    + [f"p{p}" for p in percentiles]
    + [f"prob_p{p}" for p in percentiles]
)

print(daily[output_cols].head())