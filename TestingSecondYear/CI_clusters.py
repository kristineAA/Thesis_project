import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from Predictions_S2_CV import *

res = S2_calib_results["residual_raw"].values.astype(float)

daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    m=("predicted", "size"),
)

rng = np.random.default_rng(0)

#scenario_names = ["very_low", "low", "mid", "high", "very_high"]
scenario_names = ["very_low", "mid_low", "low", "mid", "high", "mid_high", "very_high"]
k = len(scenario_names)

def cluster_probs_for_m(
    m: int,
    res: np.ndarray,
    alpha: float = 0.25,
    B: int = 20000,
    k: int = 3,
    random_state: int = 0,
):
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)

    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    central = sims[(sims >= lo) & (sims <= hi)]

    X = central.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)

    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)

    centers_sorted = centers[order]
    probs_sorted = np.bincount(labels, minlength=k)[order] / len(labels)

    out = {
        "lo_err": lo,
        "hi_err": hi,
        "n_central": len(central),
    }

    for i, name in enumerate(scenario_names):
        out[f"err_center_{name}"] = centers_sorted[i]
        out[f"prob_{name}"] = probs_sorted[i]

    return out

rows = [
    cluster_probs_for_m(
        m=int(m),
        res=res,
        alpha=0.25,
        B=20000,
        k=k,
        random_state=0,
    )
    for m in daily["m"]
]

clusters_df = pd.DataFrame(rows)
daily = pd.concat([daily.reset_index(drop=True), clusters_df.reset_index(drop=True)], axis=1)

daily["lower"] = daily["predicted"] + daily["lo_err"]
daily["upper"] = daily["predicted"] + daily["hi_err"]

for name in scenario_names:
    daily[f"center_{name}"] = daily["predicted"] + daily[f"err_center_{name}"]

output_cols = (
    ["date", "m", "n", "predicted", "lower", "upper"]
    + [f"prob_{name}" for name in scenario_names]
    + [f"center_{name}" for name in scenario_names]
)

print(daily[output_cols].head())