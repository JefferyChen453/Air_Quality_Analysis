from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROCESSED_BASE = Path("data") / "processed"


@dataclass
class DatasetBundle:
    """Container with the prepared dataframe and metadata."""

    dataframe: pd.DataFrame
    processed_dir: str


def resolve_processed_dir(state_code: int, county_code: int) -> Optional[str]:
    """Return the processed directory that matches the state/county codes."""

    prefix = f"{state_code:02d}_{county_code:03d}_"
    if not PROCESSED_BASE.exists():
        return None

    for entry in PROCESSED_BASE.iterdir():
        if entry.is_dir() and entry.name.startswith(prefix):
            return entry.name
    return None


def _slugify(column: str) -> str:
    chars = column.strip().lower()
    for old, new in [
        ("/", "_"),
        ("%", "pct"),
        ("(", ""),
        (")", ""),
        ("-", "_"),
        (",", ""),
    ]:
        chars = chars.replace(old, new)
    chars = chars.replace(" ", "_")
    while "__" in chars:
        chars = chars.replace("__", "_")
    return chars


def _load_processed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected processed file at {path}.")

    df = pd.read_csv(path)
    df.columns = [_slugify(c) for c in df.columns]

    date_col_candidates = [c for c in df.columns if c in {"date_local", "date"}]
    if not date_col_candidates:
        raise ValueError(f"Could not find a date column in {path}.")

    date_col = date_col_candidates[0]
    df["date"] = pd.to_datetime(df[date_col])
    if date_col != "date":
        df = df.drop(columns=[date_col])

    return df


def _drop_non_feature_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    cols = [c for c in columns_to_remove if c in df.columns]
    return df.drop(columns=cols, errors="ignore")


def _apply_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np # Ensure numpy is available for math operations
    temporal = pd.DataFrame(index=df.index)
    
    temporal["year"] = df["date"].dt.year
    temporal["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    
    # 1. Cyclic Month (1-12 mapped to a circle)
    temporal["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    temporal["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    
    # 2. Cyclic Day of Year (Handling leap years dynamically)
    days_in_year = df["date"].dt.is_leap_year.map({True: 366, False: 365})
    temporal["dayofyear_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / days_in_year)
    temporal["dayofyear_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / days_in_year)
    
    return pd.concat([df, temporal], axis=1)


def _apply_lag_features(df: pd.DataFrame, target_col: str = "aqi") -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    # Historical lags of the target itself (T-1, T-7, T-30)
    df[f"{target_col}_lag1"] = df[target_col].shift(1)
    df[f"{target_col}_lag7"] = df[target_col].shift(7)
    df[f"{target_col}_lag30"] = df[target_col].shift(30)
    # Rolling averages (using data available up to T-1)
    df[f"{target_col}_ma7"] = df[target_col].rolling(window=7).mean().shift(1)
    df[f"{target_col}_ma30"] = df[target_col].rolling(window=30).mean().shift(1)
    return df


def _shift_pollutant_features(df: pd.DataFrame, window_size: int = 1, keep_raw: bool = False) -> pd.DataFrame:
    """
    Shift all pollutant concentration/aqi columns by 1 to window_size days.
    This ensures that to predict AQI at day T, we only use pollutant data from [T-window, T-1].
    """
    original_cols = []
    pollutant_prefixes = ["ozone_", "so2_", "co_", "no2_", "pm25_", "pm10_", "pmc_"]
    
    cols_to_shift = [c for c in df.columns if any(c.startswith(p) for p in pollutant_prefixes)]
    
    for col in cols_to_shift:
        for lag in range(1, window_size + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        original_cols.append(col)
    
    if not keep_raw:
        df = df.drop(columns=original_cols)
    return df


def _interpolate_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply linear interpolation to pollutant columns to handle sparse sampling.
    Uses backfill for the initial gap.
    """
    pollutant_prefixes = ["ozone_", "so2_", "co_", "no2_", "pm25_", "pm10_", "pmc_"]
    cols = [c for c in df.columns if any(c.startswith(p) for p in pollutant_prefixes)]
    
    if not cols:
        return df
        
    # Interpolate linearly, then backfill the starting NaNs
    df[cols] = df[cols].interpolate(method="linear").bfill()
    return df


def build_feature_frame(
    state_code: int, county_code: int, pollutant_window: int = 1, include_targets: bool = False
) -> DatasetBundle:
    """Create a merged dataframe with engineered features for modeling."""

    processed_dir = resolve_processed_dir(state_code, county_code)
    if processed_dir is None:
        raise FileNotFoundError(
            f"Processed directory for {state_code:02d}-{county_code:03d} not found under {PROCESSED_BASE}."
        )
    base = PROCESSED_BASE / processed_dir
    def load_and_merge_subdir(subdir_name: str) -> pd.DataFrame:
        cat_dir = base / subdir_name
        if not cat_dir.exists():
            raise FileNotFoundError(f"Directory {cat_dir} does not exist.")
        
        merged_path = cat_dir / f"all_{subdir_name}_daily.csv"
        if merged_path.exists():
            return _load_processed_csv(merged_path)
        
        csv_files = [f for f in cat_dir.glob("*.csv") if not f.name.startswith("all_")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {cat_dir}")
            
        dfs = []
        for f in csv_files:
            param_prefix = f.name.replace("_daily.csv", "")
            tmp_df = _load_processed_csv(f)

            cols_to_rename = {c: f"{param_prefix}_{c}" for c in tmp_df.columns if c != "date"}
            tmp_df = tmp_df.rename(columns=cols_to_rename)
            dfs.append(tmp_df)

        merged_df = dfs[0]
        for next_df in dfs[1:]:
            merged_df = pd.merge(merged_df, next_df, on="date", how="outer")
        return merged_df
    aqi_path = base / "daily_aqi" / "aqi_daily.csv"
    aqi = _load_processed_csv(aqi_path)
    aqi = _drop_non_feature_columns(
        aqi,
        [
            "state_name",
            "county_name",
            "state_code",
            "county_code",
            "defining_parameter",
            "defining_site",
        ],
    )
    aqi = aqi[[c for c in aqi.columns if c in {"date", "aqi", "category", "number_of_sites_reporting"}]]
    aqi = aqi.rename(columns={"category": "aqi_category"})
    gases = load_and_merge_subdir("criteria_gases")
    parts = load_and_merge_subdir("particulates")
    pollutants_raw = gases.merge(parts, on="date", how="outer")
    pollutants_raw = pollutants_raw.drop(columns=[c for c in ["state_code", "county_code"] if c in pollutants_raw.columns], errors="ignore")
    pollutants_feat = _interpolate_pollutants(pollutants_raw.copy())
    pollutants_lag = _shift_pollutant_features(pollutants_feat, window_size=pollutant_window, keep_raw=False)
    meteo = load_and_merge_subdir("meteorological")
    meteo = meteo.drop(columns=[c for c in ["state_code", "county_code"] if c in meteo.columns], errors="ignore")
    df = aqi
    df = df.merge(pollutants_lag, on="date", how="left")
    if include_targets:
        df = df.merge(pollutants_raw, on="date", how="left")
    df = df.merge(meteo, on="date", how="left")
    df = _apply_temporal_features(df)
    df = _apply_lag_features(df)
    essential_lag_cols = [c for c in df.columns if c.startswith("aqi_lag") or c.startswith("aqi_ma")]
    df = df.dropna(subset=essential_lag_cols)

    return DatasetBundle(dataframe=df, processed_dir=processed_dir)



def split_by_year(
    df: pd.DataFrame,
    *,
    val_year: Optional[int] = None,
    test_year: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Split dataframe into train/val/test partitions based on calendar years."""

    years = sorted(df["date"].dt.year.unique())
    if len(years) < 3:
        raise ValueError("Need at least three distinct years for train/val/test split.")

    if test_year is None:
        test_year = years[-1]
    if val_year is None:
        val_year = years[-2]

    if val_year >= test_year:
        raise ValueError("Validation year must be earlier than test year.")

    train = df[df["date"].dt.year < val_year]
    val = df[df["date"].dt.year == val_year]
    test = df[df["date"].dt.year == test_year]

    for name, part in {"train": train, "val": val, "test": test}.items():
        if part.empty:
            raise ValueError(f"Split '{name}' is empty. Adjust val/test year selections.")

    return {"train": train, "val": val, "test": test}


POLLUTANT_TARGETS = [
    "ozone_daily_mean_conc",
    "so2_daily_mean_conc",
    "co_daily_mean_conc",
    "no2_daily_mean_conc",
    "pm25_frm_daily_mean_conc",
    "pm10_daily_mean_conc",
    "pmc_daily_mean_conc",
]


import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset for multi-output air quality prediction."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def prepare_lstm_data(
    df: pd.DataFrame, 
    lookback: int, 
    feature_cols: List[str], 
    target_cols: List[str]
):
    """
    Transform dataframe into windowed sequences for LSTM training.
    
    Returns:
        X: (N, lookback, num_features)
        y: (N, num_targets)
    """
    X_list, y_list = [], []
    
    # Note: data_utils already shifted pollutant features in build_feature_frame,
    # so the features at index i are valid for predicting targets at index i.
    
    feat_values = df[feature_cols].values
    target_values = df[target_cols].values
    
    for i in range(lookback - 1, len(df)):
        X_list.append(feat_values[i - lookback + 1 : i + 1])
        y_list.append(target_values[i])
        
    return np.array(X_list), np.array(y_list)