import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
import sys

run_dir = sys.argv[1]   # e.g. runs/xgb_20260209-104103

test_raw   = pd.read_parquet(f"{run_dir}/test_raw.parquet")
test_daily = pd.read_parquet(f"{run_dir}/test_daily.parquet")

with SummaryWriter(log_dir=run_dir) as w:
    w.add_histogram("test_raw/residuals",
                    test_raw["residual"].astype(np.float32).values, 0)

    w.add_histogram("test_daily_agg/residuals",
                    test_daily["residual"].astype(np.float32).values, 0)

    for q in (0.05,0.1,0.5,0.9,0.95):
        w.add_scalar("test_daily_agg/residual_q"+str(int(q*100)),
                     test_daily["residual"].quantile(q), 0)

print("TensorBoard updated for:", run_dir)
