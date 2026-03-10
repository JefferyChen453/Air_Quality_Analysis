import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from models.data_utils import (
    POLLUTANT_TARGETS,
    SequenceDataset,
    build_feature_frame,
    prepare_lstm_data,
    split_by_year,
)
from models.lstm_model import MultiOutputLSTM


def plot_lstm_timeseries(lstm_dir, state, county, output_path):
    lstm_dir = Path(lstm_dir)
    with open(lstm_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    lookback = meta["lookback"]
    
    # Load data
    bundle = build_feature_frame(state, county, pollutant_window=3, include_targets=True)
    df = bundle.dataframe
    
    # Load Scalers
    feature_scaler = joblib.load(lstm_dir / "feature_scaler.joblib")
    target_scaler = joblib.load(lstm_dir / "target_scaler.joblib")
    
    # Impute and Scale
    feature_meds = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(feature_meds)
    df[feature_cols] = feature_scaler.transform(df[feature_cols])
    
    # Filter Test Set
    splits = split_by_year(df, val_year=2023, test_year=2024)
    test_df = splits["test"]
    
    X_test, y_test_raw = prepare_lstm_data(test_df, lookback, feature_cols, target_cols)
    dates = test_df["date"].values[lookback-1:]
    
    # Load Model
    model = MultiOutputLSTM(len(feature_cols), len(target_cols))
    model.load_state_dict(torch.load(lstm_dir / "best_lstm.pth", map_location="cpu"))
    model.eval()
    
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    
    preds = preds_scaled * target_scaler.scale_ + target_scaler.mean_
    
    # Plotting Setup
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
    axes = axes.flatten()
    
    targets_to_plot = [
        "pm25_frm_daily_mean_conc", "pm10_daily_mean_conc",
        "ozone_daily_mean_conc", "no2_daily_mean_conc",
        "co_daily_mean_conc", "so2_daily_mean_conc"
    ]
    titles = [
        "PM2.5 (µg/m³)", "PM10 (µg/m³)",
        "Ozone (ppm)", "NO2 (ppb)",
        "CO (ppm)", "SO2 (ppb)"
    ]

    # Plot only the last 90 days for clarity
    plot_slice = slice(-90, None)
    dates_sliced = dates[plot_slice]
    preds_sliced = preds[plot_slice]
    actuals_sliced = y_test_raw[plot_slice]
    
    for i, target in enumerate(targets_to_plot):
        if target not in target_cols:
            continue
        idx = target_cols.index(target)
        
        # Extract sliced actuals and predictions for this pollutant
        y_actual = actuals_sliced[:, idx]
        p_val = preds_sliced[:, idx]
        
        # Filter NaNs for actuals
        mask = ~np.isnan(y_actual)
        valid_dates = dates_sliced[mask]
        valid_actuals = y_actual[mask]
        
        # Plot actuals as points + line
        if len(valid_actuals) > 0:
            axes[i].plot(valid_dates, valid_actuals, 'ko', label="Actual (Observed)", markersize=4, alpha=0.6)
            if len(valid_actuals) > 1:
                axes[i].plot(valid_dates, valid_actuals, 'k-', alpha=0.2)
        
        # Plot predicted as continuous line
        axes[i].plot(dates_sliced, p_val, 'r--', label="Predicted (LSTM)", alpha=0.8)
        
        axes[i].set_title(f"{titles[i]} (Target: {target})")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        plt.setp(axes[i].get_xticklabels(), rotation=30)
    
    plt.xlabel("Date")
    plt.suptitle("Multi-Output LSTM: Actual vs Predicted (Last 90 Days of 2024 Test Set)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"Saved LSTM timeseries plot to {output_path}")


def main():
    base_dir = Path("models/06_037_Los_Angeles")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    
    # 1. LSTM Timeseries
    plot_lstm_timeseries(
        base_dir / "lstm",
        6, 37,
        fig_dir / "lstm_timeseries.png"
    )


if __name__ == "__main__":
    main()
