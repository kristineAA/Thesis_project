import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from Predictions_S2_cali_CV import *

res = S2_calib_results["residual_raw"].values.astype(float)

daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    m=("predicted", "size"),
)


rng = np.random.default_rng(0)

def cluster_probs_for_m(
    m: int,
    alpha: float = 0.25,
    B: int = 20000,
    k: int = 3,
    random_state: int = 0,
):
    """
    Simulate daily total error for a day with m rows by summing m bootstrapped residuals.
    Keep only the central (1-alpha) mass (e.g. 90%), then KMeans-cluster those simulated
    errors into k clusters. Return cluster centers + probabilities (fractions).
    """
    # simulate B daily errors
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)

    # keep only central interval (within 5th..95th if alpha=0.10)
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    central = sims[(sims >= lo) & (sims <= hi)]

    # KMeans on 1D data
    X = central.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)

    centers = km.cluster_centers_.ravel()

    # order clusters by center so we can name them: low / mid / high
    order = np.argsort(centers)  # low -> high
    centers_sorted = centers[order]

    # probabilities = fraction of central sims in each cluster
    probs = np.bincount(labels, minlength=k) / len(labels)
    probs_sorted = probs[order]

    out = {
        "lo_err": lo,
        "hi_err": hi,
        "centers_low_mid_high": centers_sorted,          # error-space centers
        "p_low":  probs_sorted[0],
        "p_mid":  probs_sorted[1] if k >= 2 else np.nan,
        "p_high": probs_sorted[2] if k >= 3 else np.nan,
        "n_central": len(central),
    }
    return out

# compute for each day (based on its m)
rows = []
for m in daily["m"].astype(int):
    rows.append(cluster_probs_for_m(m, alpha=0.25, B=20000, k=3, random_state=0))

clusters_df = pd.DataFrame(rows)

# attach to daily + also convert error interval to prediction interval
daily = pd.concat([daily.reset_index(drop=True), clusters_df.reset_index(drop=True)], axis=1)
daily["lower"] = daily["predicted"] + daily["lo_err"]
daily["upper"] = daily["predicted"] + daily["hi_err"]

# OPTIONAL: cluster centers in prediction-space (not just error-space)
daily["center_low"]  = daily["predicted"] + daily["centers_low_mid_high"].str[0]
daily["center_mid"]  = daily["predicted"] + daily["centers_low_mid_high"].str[1]
daily["center_high"] = daily["predicted"] + daily["centers_low_mid_high"].str[2]

print(daily[[
    "date","m","n","predicted","lower","upper",
    "p_low","p_mid","p_high",
    "center_low","center_mid","center_high"
]].head())
