import argparse
import json
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from data_utils import build_feature_frame

##########################################################################################################
############      Run on your terminal                                                        ############
############      python models/train_rf.py --state 6 --county 37 --test-year 2025            ############
############                                                                                  ############
##########################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=int, default=6)
    parser.add_argument("--county", type=int, default=37)
    parser.add_argument("--test-year", type=int, default=2024)
    args = parser.parse_args()

    # 1. Load Data
    bundle = build_feature_frame(args.state, args.county, pollutant_window=3, include_targets=True)
    df = bundle.dataframe
    
    # Define Target and Features
    target_col = "aqi_mean" if "aqi_mean" in df.columns else "aqi"
    exclude = {target_col, "aqi_category", "date"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Fill NaNs with median (RF handles this well)
    df = df.fillna(df.median(numeric_only=True))
    
    # 2. Split Data (Train: < 2024, Test: == 2024)
    train_df = df[df["date"].dt.year < args.test_year]
    test_df = df[df["date"].dt.year == args.test_year]
    
    print(f"Training on years 2015 to {args.test_year - 1} ({len(train_df)} samples)")
    print(f"Testing on {args.test_year} ({len(test_df)} samples)")

    # 3. Train Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(train_df[feature_cols], train_df[target_col])
    
    # 4. Evaluate
    preds = rf.predict(test_df[feature_cols])
    actual = test_df[target_col].values
    
    r2 = r2_score(actual, preds)
    mae = mean_absolute_error(actual, preds)
    print(f"\nRandom Forest Results (Year {args.test_year}):")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # 5. Save Model and Plot
    out_dir = Path("models") / bundle.processed_dir / "rf_aqi"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_dir / "rf_model.joblib")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["date"], actual, label="Actual Observed", color='gray', alpha=0.5)
    plt.plot(test_df["date"], preds, label="RF Prediction", color='green', linewidth=1.5)
    plt.title(f"{args.test_year} AQI Prediction (Random Forest - Train 2015-{args.test_year-1})\nR2: {r2:.3f}, MAE: {mae:.3f}")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "rf_eval_aqi.png")
    print(f"Plot saved to {out_dir}/rf_eval_aqi.png")

if __name__ == "__main__":
    main()