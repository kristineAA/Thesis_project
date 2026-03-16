import numpy as np
import matplotlib.pyplot as plt

# data
from Preliminary_Predictions_S2 import *

# raw residuals (row-level) on test
S2_test_results["residual_raw"] = S2_test_results["n"] - S2_test_results["predicted"]

# daily sum + count
daily = S2_test_results.groupby("date", as_index=False).agg(
    n=("n", "sum"),
    predicted=("predicted", "sum"),
    m=("residual_raw", "size"),
)

band_raw = 4
daily["band_daily"] = band_raw * np.sqrt(daily["m"])
daily["lower"] = daily["predicted"] - daily["band_daily"]
daily["upper"] = daily["predicted"] + daily["band_daily"]

print(daily[["date", "n", "predicted", "lower", "upper"]].head())