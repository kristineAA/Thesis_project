from Predictions_S2_cali import *
# ================= save =================
run_dir = f"runs/xgb_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(run_dir, exist_ok=True)

final_model.save_model(f"{run_dir}/model.json")
S2_test_results.to_parquet(f"{run_dir}/S2_test_results.parquet")
S2_daily_agg.to_parquet(f"{run_dir}/S2_daily_agg.parquet")

print("Saved to:", run_dir)