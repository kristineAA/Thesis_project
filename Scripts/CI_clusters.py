import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path

# Import functions for preparing the S2 dataset and running rolling prediction windows
from Scripts.Predictions_S2_CV import prepare_S2_data, run_prediction_window

# Create a reproducible random number generator
rng = np.random.default_rng(0)

# Define scenario names
# The number of scenario names determines the number of clusters used in k-means
#scenario_names = ["low", "mid", "high"]
#scenario_names = ["very_low", "low", "mid", "high", "very_high"]
scenario_names = ["very_low", "mid_low", "low", "mid", "high", "mid_high", "very_high"]
k = len(scenario_names)

# Rolling forecast setup
FORECAST_HORIZON_DAYS = 180   # predict 6 months ahead each time
ROLL_STEP_DAYS = 90           # roll 3 months each time
TOTAL_FORECAST_DAYS = 720     # total forecast span = 2 years

# Calibration and uncertainty settings
VAL_DAYS = 90
ALPHA = 0.1
B = 20000

# Calibration and uncertainty settings
OUTPUT_DIR = Path("saved_files/rolling_3month_step_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cluster_probs_for_m(
    m: int,
    res: np.ndarray,
    alpha: float = 0.1,
    B: int = 20000,
    k: int = 3,
    random_state: int = 0,
):
    """
    Generate bootstrap-based scenario centers and probabilities
    for one day with m hospital-level observations.

    Parameters
    ----------
    m : int
        Number of hospital-level observations on the given day.
    res : np.ndarray
        Calibration residuals used for bootstrap sampling.
    alpha : float
        Fraction removed from the tails of the simulated error distribution.
    B : int
        Number of bootstrap simulations.
    k : int
        Number of scenario clusters.
    random_state : int
        Random seed used by k-means.

    Returns
    -------
    out : dict
        Dictionary containing interval error bounds, scenario error centers,
        and scenario probabilities.
    """
    # Simulate daily forecast errors by sampling m residuals with replacement
    # and summing them to match the daily aggregation level
    sims = rng.choice(res, size=(B, m), replace=True).sum(axis=1)

    # Restrict simulations to the central part of the distribution
    # This reduces the influence of extreme simulated errors before clustering
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    central = sims[(sims >= lo) & (sims <= hi)]

    # Reshape simulated errors so they can be used by sklearn's KMeans
    X = central.reshape(-1, 1)

    # Cluster the central simulated errors into k representative scenarios
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)

    # Extract cluster centers and sort them from lowest to highest demand error
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)

    centers_sorted = centers[order]

    # Estimate scenario probabilities as the relative cluster sizes
    probs_sorted = np.bincount(labels, minlength=k)[order] / len(labels)

    # Store prediction interval error bounds and number of central samples
    out = {
        "lo_err": lo,
        "hi_err": hi,
        "n_central": len(central),
    }

    # Store one error center and probability for each scenario name
    for i, name in enumerate(scenario_names):
        out[f"err_center_{name}"] = centers_sorted[i]
        out[f"prob_{name}"] = probs_sorted[i]

    return out

# Prepare the complete S2 dataset
S2 = prepare_S2_data()

# Extract all available dates in chronological order
unique_dates = np.array(sorted(S2["date"].unique()))

# Select the final two years of data as the full period to be forecasted
forecast_dates = unique_dates[-TOTAL_FORECAST_DAYS:]

all_blocks = []
block_number = 1

# Run rolling prediction blocks across the two-year forecast period
for block_start in range(0, TOTAL_FORECAST_DAYS, ROLL_STEP_DAYS):
    remaining_days = TOTAL_FORECAST_DAYS - block_start

    # Normal blocks predict 180 days ahead
    # The final block only predicts 90 days if fewer days remain
    if remaining_days <= ROLL_STEP_DAYS:
        horizon_days = ROLL_STEP_DAYS
    else:
        horizon_days = FORECAST_HORIZON_DAYS

    # Determine the date range covered by the current prediction block
    block_end = min(block_start + horizon_days, TOTAL_FORECAST_DAYS)
    block_dates = forecast_dates[block_start:block_end]
    test_start_date = pd.to_datetime(block_dates[0])
    test_end_date = pd.to_datetime(block_dates[-1])

    # Print progress information for the current block
    print("\n" + "=" * 70)
    print(
        f"Running block {block_number}: "
        f"{test_start_date.date()} to {test_end_date.date()} "
        f"({len(block_dates)} forecast days)"
    )

    # Run the prediction model for the current rolling window
    out = run_prediction_window(
        S2=S2,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        val_days=VAL_DAYS,
        n_splits=5,
    )

    # Extract calibration and test results from the prediction window
    S2_calib_results = out["S2_calib_results"]
    S2_test_results = out["S2_test_results"]

    # Use calibration residuals as the empirical error distribution
    res = S2_calib_results["residual_raw"].values.astype(float)

    # Aggregate hospital-level predictions to daily demand
    daily = S2_test_results.groupby("date", as_index=False).agg(
        n=("n", "sum"),
        predicted=("predicted", "sum"),
        m=("predicted", "size"),
    )
    # Generate scenario centers and probabilities for each forecast day
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

    # Combine daily forecasts with their corresponding scenario information
    clusters_df = pd.DataFrame(rows)
    daily = pd.concat(
        [daily.reset_index(drop=True), clusters_df.reset_index(drop=True)],
        axis=1,
    )

    # Convert error bounds into demand prediction interval bounds
    daily["lower"] = daily["predicted"] + daily["lo_err"]
    daily["upper"] = daily["predicted"] + daily["hi_err"]

    # Convert scenario error centers into absolute demand scenario values
    for name in scenario_names:
        daily[f"center_{name}"] = daily["predicted"] + daily[f"err_center_{name}"]

    # Select and order columns for the output file
    output_cols = (
        ["date", "m", "n", "predicted", "lower", "upper"]
        + [f"prob_{name}" for name in scenario_names]
        + [f"center_{name}" for name in scenario_names]
    )

    daily = daily[output_cols].copy()

    # Add block number to identify which rolling window produced each row
    daily["block"] = block_number

    # Save the current prediction block as a separate CSV file
    file_name = (
        f"predictions_block_{block_number}_"
        f"{test_start_date.strftime('%Y-%m-%d')}_to_{test_end_date.strftime('%Y-%m-%d')}.csv"
    )
    daily.to_csv(OUTPUT_DIR / file_name, index=False)

    # Store the block so all blocks can later be combined
    all_blocks.append(daily)
    block_number += 1

# Combine all rolling prediction blocks into one dataframe
final_daily = pd.concat(all_blocks, ignore_index=True).sort_values(["date", "block"])
# Save the combined rolling prediction file
final_daily.to_csv(OUTPUT_DIR / "all_rolling_predictions.csv", index=False)

# Save the combined rolling prediction file
print("\nSaved combined file:")
print(OUTPUT_DIR / "all_rolling_predictions.csv")
print(final_daily.head())