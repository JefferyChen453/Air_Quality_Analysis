import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader

from models.data_utils import (
    SequenceDataset,
    build_feature_frame,
    prepare_lstm_data,
    split_by_year,
)
from models.lstm_model import MultiOutputLSTM

def evaluate(model_dir: str, state: int, county: int):
    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    lookback = meta["lookback"]
    
    # 1. Load Data
    bundle = build_feature_frame(state, county, pollutant_window=3, include_targets=True)
    df = bundle.dataframe
    
    # 2. Load Scalers
    feature_scaler = joblib.load(model_dir / "feature_scaler.joblib")
    target_scaler = joblib.load(model_dir / "target_scaler.joblib")
    
    # 3. Apply Scaling & Imputation
    # Use training-derived medians if possible, or just the whole df for simplicity in eval
    feature_meds = df[feature_cols].median() # Ideally we'd load these from a file
    df[feature_cols] = df[feature_cols].fillna(feature_meds)
    df[feature_cols] = feature_scaler.transform(df[feature_cols])
    
    # 4. Filter Test Set
    splits = split_by_year(df, val_year=2023, test_year=2024)
    test_df = splits["test"]
    
    X_test, y_test_raw = prepare_lstm_data(test_df, lookback, feature_cols, target_cols)
    
    # 5. Load Model
    input_dim = len(feature_cols)
    output_dim = len(target_cols)
    model = MultiOutputLSTM(input_dim, output_dim)
    model.load_state_dict(torch.load(model_dir / "best_lstm.pth", map_location="cpu"))
    model.eval()
    
    # 6. Predict
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    
    # 7. Inverse Transform
    # preds_scaled are the model outputs (scaled)
    # y_test_raw are the ground truth from prepare_lstm_data (which are raw in this script)
    preds = preds_scaled * target_scaler.scale_ + target_scaler.mean_
    y_test = y_test_raw # prepare_lstm_data returns raw values from test_df as we didn't scale them here
    
    # 8. Calculate Metrics per Pollutant
    results = {}
    for i, col in enumerate(target_cols):
        # Handle potential NaNs in ground truth (mask them out for metric calculation)
        mask = ~np.isnan(y_test[:, i])
        if mask.any():
            r2 = r2_score(y_test[mask, i], preds[mask, i])
            mae = mean_absolute_error(y_test[mask, i], preds[mask, i])
            results[col] = {"r2": float(r2), "mae": float(mae)}
        else:
            results[col] = {"r2": None, "mae": None}
            
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    # Path to the specific LA model dir
    model_path = "models/06_037_Los_Angeles/lstm"
    evaluate(model_path, 6, 37)
