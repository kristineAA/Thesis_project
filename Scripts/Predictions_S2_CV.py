import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Scripts.CleanDataPreliminary import dataset_filtered


# ================= data preparation =================
def prepare_S2_data():
    S2 = dataset_filtered.copy()

    S2["date"] = pd.to_datetime(S2["date"])
    S2["day_of_week"] = S2["date"].dt.dayofweek
    S2["is_weekend"] = (S2["day_of_week"] >= 5).astype(int)
    S2["month"] = S2["date"].dt.month
    S2["year"] = S2["date"].dt.year
    S2["day_of_month"] = S2["date"].dt.day

    S2 = S2[
        [
            "date",
            "day_of_week",
            "is_weekend",
            "month",
            "year",
            "day_of_month",
            "is_holiday",
            "name",
            "size",
            "Hospital ID",
            "y_lat",
            "x_lon",
            "region",
            "n",
        ]
    ].copy()

    S2["name"] = S2["name"].astype("category")
    S2["region"] = S2["region"].astype("category")
    S2 = S2.sort_values("date").reset_index(drop=True)

    return S2


FEATURE_COLS = [
    "day_of_week",
    "is_weekend",
    "month",
    "year",
    "day_of_month",
    "is_holiday",
    "name",
    "size",
    "Hospital ID",
    "y_lat",
    "x_lon",
    "region",
]
TARGET_COL = "n"


# ================= date-based CV helper =================
def make_date_folds(df, n_splits=5):
    unique_dates = np.array(sorted(df["date"].unique()))
    n_dates = len(unique_dates)

    if n_splits + 1 > n_dates:
        raise ValueError("Too many splits for number of unique dates")

    fold_size = n_dates // (n_splits + 1)
    remainder = n_dates % (n_splits + 1)

    blocks = []
    start = 0
    for i in range(n_splits + 1):
        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        blocks.append(unique_dates[start:end])
        start = end

    folds = []
    for i in range(1, len(blocks)):
        train_dates = np.concatenate(blocks[:i])
        cv_dates = blocks[i]

        train_idx = df.index[df["date"].isin(train_dates)].to_numpy()
        cv_idx = df.index[df["date"].isin(cv_dates)].to_numpy()

        folds.append((train_idx, cv_idx))

    return folds


# ================= rolling training function =================
def run_prediction_window(
    S2: pd.DataFrame,
    test_start_date,
    test_end_date,
    val_days: int = 90,
    n_splits: int = 5,
):
    test_start_date = pd.to_datetime(test_start_date)
    test_end_date = pd.to_datetime(test_end_date)
    cutoff_val = test_start_date - pd.Timedelta(days=val_days)

    train_data_S2 = S2[S2["date"] < cutoff_val].copy()
    val_data_S2 = S2[(S2["date"] >= cutoff_val) & (S2["date"] < test_start_date)].copy()
    test_data_S2 = S2[(S2["date"] >= test_start_date) & (S2["date"] <= test_end_date)].copy()

    if train_data_S2.empty:
        raise ValueError("Training set is empty for this window.")
    if val_data_S2.empty:
        raise ValueError("Validation set is empty for this window.")
    if test_data_S2.empty:
        raise ValueError("Test set is empty for this window.")

    X_train = train_data_S2[FEATURE_COLS]
    y_train = train_data_S2[TARGET_COL].values

    X_val = val_data_S2[FEATURE_COLS]
    y_val = val_data_S2[TARGET_COL].values

    X_test = test_data_S2[FEATURE_COLS]
    y_test = test_data_S2[TARGET_COL].values

    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    date_folds = make_date_folds(train_data_S2, n_splits=n_splits)

    param_grid = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.1, 0.2, 0.3],
        "n_estimators": [300, 800, 1500],
    }

    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "seed": 43,
    }

    best = {
        "mse": float("inf"),
        "params": None,
        "best_iteration": None,
    }

    best_oof_parts = None

    for md in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            for n_est in param_grid["n_estimators"]:
                params = dict(base_params, max_depth=md, learning_rate=lr)

                fold_mses = []
                fold_best_iterations = []
                oof_parts = []

                for fold_num, (train_idx, cv_idx) in enumerate(date_folds, start=1):
                    fold_train = train_data_S2.loc[train_idx]
                    fold_cv = train_data_S2.loc[cv_idx]

                    X_tr = fold_train[FEATURE_COLS]
                    y_tr = fold_train[TARGET_COL].values
                    X_cv = fold_cv[FEATURE_COLS]
                    y_cv = fold_cv[TARGET_COL].values

                    dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
                    dcv = xgb.DMatrix(X_cv, y_cv, enable_categorical=True)

                    model = xgb.train(
                        params=params,
                        dtrain=dtr,
                        num_boost_round=n_est,
                        evals=[(dcv, "cv")],
                        early_stopping_rounds=50,
                        verbose_eval=False,
                    )

                    pred_cv = model.predict(dcv)
                    mse_cv = mean_squared_error(y_cv, pred_cv)

                    fold_mses.append(mse_cv)
                    fold_best_iterations.append(model.best_iteration + 1)

                    fold_res = fold_cv[["date", TARGET_COL]].copy()
                    fold_res["predicted"] = pred_cv
                    fold_res["residual_raw"] = fold_res[TARGET_COL] - fold_res["predicted"]
                    fold_res["fold"] = fold_num
                    oof_parts.append(fold_res)

                avg_mse = np.mean(fold_mses)
                avg_best_iteration = int(np.round(np.mean(fold_best_iterations)))

                print(
                    f"Window {test_start_date.date()} to {test_end_date.date()} | "
                    f"md={md}, lr={lr}, n_est={n_est}, "
                    f"cv_mse={avg_mse:.4f}, best_iter≈{avg_best_iteration}"
                )

                if avg_mse < best["mse"]:
                    best.update(
                        mse=avg_mse,
                        params={
                            "max_depth": md,
                            "learning_rate": lr,
                            "n_estimators": n_est,
                        },
                        best_iteration=avg_best_iteration,
                    )
                    best_oof_parts = oof_parts

    print("\nBest params from date-based CV:")
    print(best)

    S2_calib_results = (
        pd.concat(best_oof_parts, axis=0)
        .sort_values("date")
        .reset_index(drop=True)
    )

    final_params = dict(
        base_params,
        max_depth=best["params"]["max_depth"],
        learning_rate=best["params"]["learning_rate"],
    )

    final_model = xgb.train(
        params=final_params,
        dtrain=dtrain,
        num_boost_round=best["params"]["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    plt.figure(figsize=(12, 8))  # make figure larger

    xgb.plot_importance(final_model)

    plt.tight_layout()           # automatically fix spacing
    plt.subplots_adjust(left=0.3)  # extra room for long labels

    plt.savefig("feature_importance.png", bbox_inches="tight")
    plt.close()

    # plot random tree
    xgb.plot_tree(final_model, num_trees=1)
    plt.savefig("random_tree.png", bbox_inches="tight")
    plt.close()

    val_pred = final_model.predict(dval)
    test_pred = np.round(final_model.predict(dtest))

    results_val = {
        "MAE": mean_absolute_error(y_val, val_pred),
        "MSE": mean_squared_error(y_val, val_pred),
        "RMSE": np.sqrt(mean_squared_error(y_val, val_pred)),
        "STD": np.std(y_val - val_pred),
    }

    S2_test_results = test_data_S2.copy()
    S2_test_results["predicted"] = test_pred
    S2_test_results["residual"] = S2_test_results["n"] - S2_test_results["predicted"]

    results_test = {
        "MAE": mean_absolute_error(y_test, test_pred),
        "MSE": mean_squared_error(y_test, test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "STD": np.std(y_test - test_pred),
    }

    S2_daily_agg = S2_test_results.groupby("date", as_index=False).agg(
        n=("n", "sum"),
        predicted=("predicted", "sum"),
    )
    S2_daily_agg["residual"] = S2_daily_agg["n"] - S2_daily_agg["predicted"]

    print("\nValidation results:")
    print(results_val)

    print("\nTest results:")
    print(results_test)

    print("\nCalibration residual sample:")
    print(S2_calib_results.head())

    return {
        "S2_calib_results": S2_calib_results,
        "S2_test_results": S2_test_results,
        "S2_daily_agg": S2_daily_agg,
        "results_val": results_val,
        "results_test": results_test,
        "best": best,
    }


if __name__ == "__main__":
    S2 = prepare_S2_data()

    unique_dates = np.array(sorted(S2["date"].unique()))
    test_days = 729
    val_days = 90

    forecast_dates = unique_dates[-test_days:]
    test_start_date = forecast_dates[0]
    test_end_date = forecast_dates[-1]

    run_prediction_window(
        S2=S2,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        val_days=val_days,
        n_splits=5,
    )