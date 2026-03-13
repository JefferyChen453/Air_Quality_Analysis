import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from data_utils import build_feature_frame

def main():
    state = 6
    county = 37

    print("Loading data to reconstruct feature names...")
    bundle = build_feature_frame(state, county, pollutant_window=3, include_targets=True)
    df = bundle.dataframe
    
    target_col = "aqi_mean" if "aqi_mean" in df.columns else "aqi"
    exclude = {target_col, "aqi_category", "date"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    model_dir = Path("models") / bundle.processed_dir / "rf_aqi"
    model_path = model_dir / "rf_model.joblib"
    
    if not model_path.exists():
        print(f"Error: Could not find model at {model_path}")
        return
        
    print("Loading Random Forest model...")
    rf = joblib.load(model_path)
    
    importances = rf.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    })
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(20)
    
    print("\n--- Top 10 Most Important Features ---")
    print(fi_df.head(10).to_string(index=False))

    plt.figure(figsize=(12, 8))
    plt.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color='teal')
    plt.xlabel("Random Forest Feature Importance (Gini Importance)")
    plt.ylabel("Features")
    plt.title("Top 20 Features Driving AQI Prediction in LA County")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = model_dir / "rf_feature_importance.png"
    plt.savefig(plot_path)
    print(f"\n Feature importance plot successfully saved to:\n{plot_path}")

if __name__ == "__main__":
    main()