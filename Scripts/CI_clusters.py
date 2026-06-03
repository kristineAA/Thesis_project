import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path

from Scripts.Predictions_S2_CV import prepare_S2_data, run_prediction_window


rng = np.random.default_rng(0)

#scenario_names = ["low", "mid", "high"]
#scenario_names = ["very_low", "low", "mid", "high", "very_high"]
scenario_names = ["very_low", "mid_low", "low", "mid", "high", "mid_high", "very_high"]
k = len(scenario_names)

FORECAST_HORIZON_DAYS = 180   # predict 6 months ahead each time
ROLL_STEP_DAYS = 90           # roll 3 months each time
TOTAL_FORECAST_DAYS = 720     # total forecast span = 2 years
VAL_DAYS = 90
ALPHA = 0.25
B = 20000

OUTPUT_DIR = Path("saved_files/rolling_3month_step_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


S2 = prepare_S2_data()
unique_dates = np.array(sorted(S2["date"].unique()))

# the full 2-year period we want to cover
forecast_dates = unique_dates[-TOTAL_FORECAST_DAYS:]

all_blocks = []
block_number = 1

for block_start in range(0, TOTAL_FORECAST_DAYS, ROLL_STEP_DAYS):
    remaining_days = TOTAL_FORECAST_DAYS - block_start

    # Normal runs predict 180 days ahead, but the final run predicts only 90 days
    if remaining_days <= ROLL_STEP_DAYS:
        horizon_days = ROLL_STEP_DAYS
    else:
        horizon_days = FORECAST_HORIZON_DAYS

    block_end = min(block_start + horizon_days, TOTAL_FORECAST_DAYS)

    block_dates = forecast_dates[block_start:block_end]
    test_start_date = pd.to_datetime(block_dates[0])
    test_end_date = pd.to_datetime(block_dates[-1])

    print("\n" + "=" * 70)
    print(
        f"Running block {block_number}: "
        f"{test_start_date.date()} to {test_end_date.date()} "
        f"({len(block_dates)} forecast days)"
    )

    out = run_prediction_window(
        S2=S2,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        val_days=VAL_DAYS,
        n_splits=5,
    )

    S2_calib_results = out["S2_calib_results"]
    S2_test_results = out["S2_test_results"]

    res = S2_calib_results["residual_raw"].values.astype(float)

    daily = S2_test_results.groupby("date", as_index=False).agg(
        n=("n", "sum"),
        predicted=("predicted", "sum"),
        m=("predicted", "size"),
    )

    rows = [
        cluster_probs_for_m(
            m=int(m),
            res=res,
            alpha=ALPHA,
            B=B,
            k=k,
            random_state=0,
        )
        for m in daily["m"]
    ]

    clusters_df = pd.DataFrame(rows)
    daily = pd.concat(
        [daily.reset_index(drop=True), clusters_df.reset_index(drop=True)],
        axis=1,
    )

    daily["lower"] = daily["predicted"] + daily["lo_err"]
    daily["upper"] = daily["predicted"] + daily["hi_err"]

    for name in scenario_names:
        daily[f"center_{name}"] = daily["predicted"] + daily[f"err_center_{name}"]

    output_cols = (
        ["date", "m", "n", "predicted", "lower", "upper"]
        + [f"prob_{name}" for name in scenario_names]
        + [f"center_{name}" for name in scenario_names]
    )

    daily = daily[output_cols].copy()
    daily["block"] = block_number

    file_name = (
        f"predictions_block_{block_number}_"
        f"{test_start_date.strftime('%Y-%m-%d')}_to_{test_end_date.strftime('%Y-%m-%d')}.csv"
    )
    daily.to_csv(OUTPUT_DIR / file_name, index=False)

    all_blocks.append(daily)
    block_number += 1

final_daily = pd.concat(all_blocks, ignore_index=True).sort_values(["date", "block"])
final_daily.to_csv(OUTPUT_DIR / "all_rolling_predictions.csv", index=False)

print("\nSaved combined file:")
print(OUTPUT_DIR / "all_rolling_predictions.csv")
print(final_daily.head())