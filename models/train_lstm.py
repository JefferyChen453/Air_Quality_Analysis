import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models.data_utils import (
    POLLUTANT_TARGETS,
    SequenceDataset,
    build_feature_frame,
    prepare_lstm_data,
    split_by_year,
)
from models.lstm_model import MultiOutputLSTM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=int, default=6, help="State code.")
    parser.add_argument("--county", type=int, default=37, help="County code.")
    parser.add_argument("--lookback", type=int, default=14, help="Days of history to look back.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-year", type=int, default=2023)
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


class MaskedMSELoss(nn.Module):
    """Loss function that ignores NaNs in the ground truth."""

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        diff = (pred[mask] - target[mask]) ** 2
        return diff.mean() if mask.any() else torch.tensor(0.0, requires_grad=True)



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, out_dir):
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f}")
            
        if scheduler is not None:
            scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), out_dir / "best_lstm.pth")

    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    bundle = build_feature_frame(args.state, args.county, pollutant_window=3, include_targets=True)
    df = bundle.dataframe
    
    # Identify feature columns (exclude date and all potential targets)
    exclude_targets = set(POLLUTANT_TARGETS) | {"aqi", "aqi_category", "date"}
    feature_cols = [c for c in df.columns if c not in exclude_targets]
    
    # 2. Scaling
    train_df = df[df["date"].dt.year < args.val_year]
    
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_cols])
    
    target_scaler = StandardScaler()
    target_meds = train_df[POLLUTANT_TARGETS].median()
    target_scaler.fit(train_df[POLLUTANT_TARGETS].fillna(target_meds))
    
    # Fill NaNs in features (Time-Series Friendly)
    df[feature_cols] = df[feature_cols].interpolate(method='linear', limit=7)
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    # Apply scaling to whole df
    df[feature_cols] = feature_scaler.transform(df[feature_cols])
    df[POLLUTANT_TARGETS] = (df[POLLUTANT_TARGETS] - target_scaler.mean_) / target_scaler.scale_
    
    # 3. Split
    splits = split_by_year(df, val_year=args.val_year, test_year=args.test_year)
    
    # 4. Create sequences
    data_pools = {}
    for name, part in splits.items():
        X, y = prepare_lstm_data(part, args.lookback, feature_cols, POLLUTANT_TARGETS)
        data_pools[name] = DataLoader(SequenceDataset(X, y), batch_size=args.batch_size, shuffle=(name == "train"))
        
    # 5. Model Initialization
    input_dim = len(feature_cols)
    output_dim = len(POLLUTANT_TARGETS)
    model = MultiOutputLSTM(input_dim, output_dim, hidden_size=args.hidden_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = MaskedMSELoss()
    
    out_dir = Path(args.output_dir) if args.output_dir else Path("models") / bundle.processed_dir / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 6. Train
    print(f"Starting LSTM Training for {bundle.processed_dir}...")
    train_model(model, data_pools["train"], data_pools["val"], criterion, optimizer, scheduler, args.epochs, device, out_dir)
    
    # 7. Final Save
    joblib.dump(feature_scaler, out_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, out_dir / "target_scaler.joblib")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "target_cols": POLLUTANT_TARGETS,
            "lookback": args.lookback
        }, f, indent=2)

if __name__ == "__main__":
    main()
