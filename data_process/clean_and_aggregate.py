"""
Clean and aggregate EPA Daily Summary data for a specific state code and county code.

Processes categories:
  - Criteria Gases (O3, SO2, CO, NO2)
  - Particulates (PM2.5 FRM, PM2.5 non-FRM, PM10, PMc)
  - Meteorological (Wind, Temperature, Pressure, RH/Dewpoint)
  - Toxics, Precursors, and Lead (HAPS, VOCS, Lead, NONOxNOy)
  - Daily AQI by County (just filter, already county-level)

Output data tree structure:
  ./data/processed/{state_code}_{county_code}_{county_name}/
      ├── criteria_gases/
      ├── particulates/
      ├── meteorological/
      ├── toxics_precursors_lead/
      └── daily_aqi/

Usage:
    uv run data_process/clean_aggregate_county.py --state 6 --county 37
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


# Config
RAW_BASE = os.path.join(".", "data", "raw", "daily_summary")

CATEGORIES = {
    "criteria_gases": {
        "raw_subdir": "criteria_gases",
        "out_subdir": "criteria_gases",
        "merge_name": "all_criteria_gases_daily",
        "params": {
            "44201": {"out_name": "ozone", "label": "Ozone (O3)",
                      "preferred_duration": None, "unit": "ppm",
                      "is_multi_param": False},
            "42401": {"out_name": "so2", "label": "SO2",
                      "preferred_duration": "1 HOUR", "unit": "ppb",
                      "is_multi_param": False},
            "42101": {"out_name": "co", "label": "CO",
                      "preferred_duration": "1 HOUR", "unit": "ppm",
                      "is_multi_param": False},
            "42602": {"out_name": "no2", "label": "NO2",
                      "preferred_duration": "1 HOUR", "unit": "ppb",
                      "is_multi_param": False},
        },
    },
    "particulates": {
        "raw_subdir": "particulates",
        "out_subdir": "particulates",
        "merge_name": "all_particulates_daily",
        "params": {
            "88101": {"out_name": "pm25_frm", "label": "PM2.5 FRM/FEM",
                      "preferred_duration": "24 HOUR", "unit": "ug/m3 LC",
                      "is_multi_param": False},
            "88502": {"out_name": "pm25_nonfrm", "label": "PM2.5 non-FRM",
                      "preferred_duration": "24 HOUR", "unit": "ug/m3 LC",
                      "is_multi_param": False},
            "81102": {"out_name": "pm10", "label": "PM10 Mass",
                      "preferred_duration": "24 HOUR", "unit": "ug/m3 SC",
                      "is_multi_param": False},
            "86101": {"out_name": "pmc", "label": "PMc Mass",
                      "preferred_duration": "24 HOUR", "unit": "ug/m3 LC",
                      "is_multi_param": False},
        },
    },
    "meteorological": {
        "raw_subdir": "meteorological",
        "out_subdir": "meteorological",
        "merge_name": "all_meteorological_daily",
        "params": {
            "WIND": {"out_name": "wind", "label": "Wind",
                     "preferred_duration": "1 HOUR", "unit": "Knots",
                     "is_multi_param": True},
            "TEMP": {"out_name": "temperature", "label": "Temperature",
                     "preferred_duration": "1 HOUR", "unit": "Deg F",
                     "is_multi_param": True},
            "PRESS": {"out_name": "pressure", "label": "Barometric Pressure",
                      "preferred_duration": "1 HOUR", "unit": "Millibars",
                      "is_multi_param": True},
            "RH_DP": {"out_name": "rh_dewpoint", "label": "RH and Dewpoint",
                      "preferred_duration": "1 HOUR", "unit": "various",
                      "is_multi_param": True},
        },
    },
    "toxics": {
        "raw_subdir": "toxics_precursors_lead",
        "out_subdir": "toxics_precursors_lead",
        "merge_name": "all_toxics_daily",
        "params": {
            "HAPS": {"out_name": "haps", "label": "Hazardous Air Pollutants",
                     "preferred_duration": "24 HOUR", "unit": "various",
                     "is_multi_param": True},
            "VOCS": {"out_name": "vocs", "label": "Volatile Organic Compounds",
                     "preferred_duration": "24 HOUR", "unit": "various",
                     "is_multi_param": True},
            "NONOxNOy": {"out_name": "no_nox_noy", "label": "NO/NOx/NOy Precursors",
                         "preferred_duration": "1 HOUR", "unit": "ppb",
                         "is_multi_param": True},
            "LEAD": {"out_name": "lead", "label": "Lead",
                     "preferred_duration": "24 HOUR", "unit": "ug/m3 LC",
                     "is_multi_param": False},
        },
    },
    "aqi": {
        "raw_subdir": "daily_aqi",
        "out_subdir": "daily_aqi",
        "merge_name": None,
        "params": {
            "aqi_by_county": {"out_name": "aqi", "label": "Daily AQI by County",
                              "preferred_duration": None, "unit": "AQI",
                              "is_multi_param": False, "is_aqi": True},
        },
    },
}

CATEGORY_ALIASES = {
    "gases": "criteria_gases", "gas": "criteria_gases",
    "pm": "particulates",
    "met": "meteorological", "meteo": "meteorological",
    "toxic": "toxics",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and aggregate EPA data by county.")
    parser.add_argument("--state", type=int, required=True, help="state code")
    parser.add_argument("--county", type=int, required=True, help="county code")
    parser.add_argument("--year", type=int, nargs="+", default=[2025], help="Data year(s) (e.g., 2025, or 2015 2025 for a range)")
    parser.add_argument("--category", nargs="*", default=None,
                        help="Categories to process (default to process all categories)")
    return parser.parse_args()


def resolve_categories(cat_args):
    if cat_args is None or "all" in (cat_args or []):
        return list(CATEGORIES.keys())
    resolved = []
    for c in cat_args:
        cl = c.lower()
        if cl in CATEGORIES:
            resolved.append(cl)
        elif cl in CATEGORY_ALIASES:
            resolved.append(CATEGORY_ALIASES[cl])
    return list(dict.fromkeys(resolved))


# Data loading
def load_raw_csv_years(raw_subdir, param_code, years):
    dfs = []
    for year in years:
        filename = f"daily_{param_code}_{year}.csv"
        filepath = os.path.join(RAW_BASE, raw_subdir, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: Not found: {filepath}")
            continue
        print(f"  Loading: {filename}")
        df = pd.read_csv(filepath, low_memory=False)
        dfs.append(df)
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows combined: {len(combined_df):,}")
    return combined_df


def filter_county(df, state_code, county_code):
    mask = (df["State Code"] == state_code) & (df["County Code"] == county_code)
    result = df[mask].copy()
    print(f"  County filter -> {len(result):,} rows")
    return result


def get_county_name(df):
    if "County Name" in df.columns and len(df) > 0:
        return df["County Name"].iloc[0].strip().replace(" ", "_")
    return None


# Cleaning
def clean_records(df, param_config):
    """
    Clean steps:
        - Event Type: exclude "Events Excluded" / "Concurred Events Excluded" (duplicates)
        - Sample Duration: keep appropriate duration (8-hr for O3, 1-hr for other gases, 24-hr for PM)
        - Pollutant Standard: keep most recent if multiple
    """

    if "Event Type" in df.columns:
        vals = df["Event Type"].unique()
        print(f"  Event Type values: {list(vals)}")
        exclude = ["Events Excluded", "Concurred Events Excluded"]
        mask = df["Event Type"].isin(exclude)
        if 0 < mask.sum() < len(df):
            df = df[~mask].copy()
            print(f"  -> Removed {mask.sum()} excluded-event rows, keeping {len(df):,}")
        else:
            print(f"  -> Keeping all {len(df):,} rows")

    if "Sample Duration" in df.columns:
        durations = df["Sample Duration"].unique()
        print(f"  Sample Duration values: {list(durations)}")
        preferred = param_config.get("preferred_duration")

        if preferred is not None:
            mask = df["Sample Duration"] == preferred
            if mask.sum() > 0:
                before = len(df)
                df = df[mask].copy()
                print(f"  -> Duration '{preferred}': {len(df):,} (removed {before - len(df)})")
            else:
                mc = df["Sample Duration"].value_counts().index[0]
                df = df[df["Sample Duration"] == mc].copy()
                print(f"  -> '{preferred}' not found, using '{mc}': {len(df):,}")
        elif not param_config.get("is_multi_param", False):
            # Ozone: prefer 8-hour
            eight = [d for d in durations if "8" in str(d).upper()]
            if eight:
                before = len(df)
                df = df[df["Sample Duration"] == eight[0]].copy()
                print(f"  -> Auto '{eight[0]}': {len(df):,} (removed {before - len(df)})")
            else:
                mc = df["Sample Duration"].value_counts().index[0]
                df = df[df["Sample Duration"] == mc].copy()
                print(f"  -> Most common '{mc}': {len(df):,}")

    if "Pollutant Standard" in df.columns:
        stds = df["Pollutant Standard"].dropna().unique()
        print(f"  Pollutant Standard values: {list(stds)}")
        if len(stds) > 1:
            keep = sorted(stds)[-1]
            before = len(df)
            df = df[df["Pollutant Standard"] == keep].copy()
            print(f"  -> Keeping '{keep}': {len(df):,} (removed {before - len(df)})")

    return df



def resolve_poc(df):
    before = len(df)
    df = df.sort_values("POC").groupby(["Date Local", "Site Num"]).first().reset_index()
    removed = before - len(df)
    if removed > 0:
        print(f"  POC dedup: removed {removed}, keeping {len(df):,}")
    return df


def aggregate_single_param(df, param_config):
    """
    Aggregate a single-parameter file to county daily level:
        - Arithmetic Mean: mean, max, median, std
        - 1st Max Value: max
        - Observation Count: sum
        - AQI: mean, max
    """
    agg = {"Arithmetic Mean": ["mean", "max", "median", "std"],
           "1st Max Value": ["max"]}
    if "Observation Count" in df.columns:
        agg["Observation Count"] = ["sum"]
    if "AQI" in df.columns and df["AQI"].notna().sum() > 0:
        agg["AQI"] = ["mean", "max"]

    county = df.groupby("Date Local").agg(agg)
    county.columns = ["_".join(c) for c in county.columns]
    county["station_count"] = df.groupby("Date Local")["Site Num"].nunique()

    rn = {"Arithmetic Mean_mean": "daily_mean_conc",
          "Arithmetic Mean_max": "daily_max_site_mean",
          "Arithmetic Mean_median": "daily_median_conc",
          "Arithmetic Mean_std": "daily_std_conc",
          "1st Max Value_max": "daily_peak_conc"}
    if "Observation Count_sum" in county.columns:
        rn["Observation Count_sum"] = "total_observations"
    if "AQI_mean" in county.columns:
        rn["AQI_mean"] = "aqi_mean"
        rn["AQI_max"] = "aqi_max"

    county = county.rename(columns=rn).reset_index()
    county["Date Local"] = pd.to_datetime(county["Date Local"], format="mixed")
    county = county.sort_values("Date Local").reset_index(drop=True)
    county["parameter"] = param_config["out_name"]
    county["unit"] = param_config["unit"]

    num_cols = county.select_dtypes(include=[np.number]).columns
    county[num_cols] = county[num_cols].round(4)
    return county


def aggregate_multi_param(df, param_config):
    if "Parameter Name" not in df.columns:
        return aggregate_single_param(df, param_config)

    params = df["Parameter Name"].unique()
    print(f"  Sub-parameters ({len(params)}): {list(params[:10])}"
          f"{'...' if len(params) > 10 else ''}")

    # For huge multi-param files, keep top 15 by data volume
    if len(params) > 15:
        top = df["Parameter Name"].value_counts().head(15).index.tolist()
        print(f"  -> Keeping top 15 by data volume")
        df = df[df["Parameter Name"].isin(top)].copy()
        params = top

    pieces = []
    for pname in params:
        dp = df[df["Parameter Name"] == pname].copy()
        if len(dp) == 0:
            continue

        # POC deduplication per sub-parameter
        dp = dp.sort_values("POC").groupby(["Date Local", "Site Num"]).first().reset_index()

        daily = dp.groupby("Date Local").agg(
            mean=("Arithmetic Mean", "mean"),
            peak=("1st Max Value", "max"),
            stations=("Site Num", "nunique"),
        ).reset_index()

        # Safe column name
        safe = (pname.lower().replace(" ", "_").replace("(", "").replace(")", "")
                .replace(",", "").replace("/", "_").replace("-", "_").replace(".", ""))
        if len(safe) > 30:
            safe = safe[:30]

        daily = daily.rename(columns={
            "mean": f"{safe}_mean", "peak": f"{safe}_peak",
            "stations": f"{safe}_stations"})
        pieces.append(daily)

    if not pieces:
        return None

    merged = pieces[0]
    for p in pieces[1:]:
        merged = pd.merge(merged, p, on="Date Local", how="outer")

    merged["Date Local"] = pd.to_datetime(merged["Date Local"], format="mixed")
    merged = merged.sort_values("Date Local").reset_index(drop=True)
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].round(4)
    return merged


# AQI special processing
def process_aqi(years, state_code, county_code):
    """
    Daily AQI by County: load and filter (already county-level so no need to agg by sites):
    """
    print(f"\n{'─'*60}")
    print(f"Processing: Daily AQI by County for years {years}")
    print(f"{'─'*60}")

    df = load_raw_csv_years("daily_aqi", "aqi_by_county", years)
    if df is None:
        return None, None

    cols = df.columns.tolist()
    print(f"  Columns: {cols}")

    state_col = next((c for c in cols if c.strip().replace(" ", "").lower() == "statecode"), None)
    county_col = next((c for c in cols if c.strip().replace(" ", "").lower() == "countycode"), None)

    if state_col is None:
        state_col = next((c for c in cols if "state" in c.lower() and "code" in c.lower()), None)
    if county_col is None:
        county_col = next((c for c in cols if "county" in c.lower() and "code" in c.lower()), None)

    if state_col is None or county_col is None:
        print(f"  ERROR: Cannot find State/County Code columns in: {cols}")
        print(f"  Detected state_col={state_col}, county_col={county_col}")
        return None, None

    print(f"  Using columns: state='{state_col}', county='{county_col}'")

    df[state_col] = pd.to_numeric(df[state_col], errors="coerce")
    df[county_col] = pd.to_numeric(df[county_col], errors="coerce")

    mask = (df[state_col] == state_code) & (df[county_col] == county_code)
    result = df[mask].copy()
    print(f"  County filter -> {len(result):,} rows")

    if len(result) == 0:
        print(f"  No AQI data for this county!")
        return None, None

    name_col = next((c for c in result.columns if "county" in c.lower() and "name" in c.lower()), None)
    county_name = None
    if name_col:
        county_name = result[name_col].iloc[0].strip().replace(" ", "_")

    date_col = next((c for c in result.columns if "date" in c.lower()), None)
    if date_col:
        result[date_col] = pd.to_datetime(result[date_col], format="mixed")
        result = result.sort_values(date_col).reset_index(drop=True)
        print(f"  Date range: {result[date_col].min().date()} to {result[date_col].max().date()}")

    return result, county_name


# Per-param pipeline
def process_one_param(raw_subdir, param_code, param_config, state_code, county_code, years):
    print(f"\n{'─'*60}")
    print(f"Processing: {param_config['label']} ({param_code}) for years {years}")
    print(f"{'─'*60}")

    df = load_raw_csv_years(raw_subdir, param_code, years)
    if df is None or len(df) == 0:
        return None, None

    df_county = filter_county(df, state_code, county_code)
    if len(df_county) == 0:
        print(f"  No data for this county!")
        return None, None

    county_name = get_county_name(df_county)
    df_clean = clean_records(df_county, param_config)
    if len(df_clean) == 0:
        print(f"  No data after cleaning!")
        return None, county_name

    is_multi = param_config.get("is_multi_param", False)
    if is_multi:
        result = aggregate_multi_param(df_clean, param_config)
    else:
        df_site = resolve_poc(df_clean)
        # Print site info
        if "Local Site Name" in df_site.columns:
            sites = df_site[["Site Num", "Local Site Name"]].drop_duplicates()
        else:
            sites = df_site[["Site Num"]].drop_duplicates()
        print(f"  Sites ({len(sites)}):")
        for _, row in sites.head(10).iterrows():
            print(f"    {row['Site Num']:>5}: {row.get('Local Site Name', 'N/A')}")
        if len(sites) > 10:
            print(f"    ... and {len(sites) - 10} more")
        result = aggregate_single_param(df_site, param_config)

    if result is None or len(result) == 0:
        return None, county_name

    print(f"  Result: {len(result)} daily records")
    print(f"  Date range: {result['Date Local'].min().date()} "
          f"to {result['Date Local'].max().date()}")
    return result, county_name


# Merge
def merge_results(results, is_any_multi=False):
    merged = None
    for name, df in results.items():
        if is_any_multi:
            slim = df.copy()
        else:
            keep = ["Date Local", "daily_mean_conc", "daily_peak_conc", "station_count"]
            if "aqi_max" in df.columns:
                keep.append("aqi_max")
            available = [c for c in keep if c in df.columns]
            slim = df[available].copy()
            slim = slim.rename(columns={c: f"{name}_{c}" for c in available if c != "Date Local"})

        if merged is None:
            merged = slim
        else:
            merged = pd.merge(merged, slim, on="Date Local", how="outer")

    if merged is not None:
        merged = merged.sort_values("Date Local").reset_index(drop=True)
    return merged


def main():
    args = parse_args()
    
    years = args.year
    if len(years) == 2:
        years = list(range(years[0], years[1] + 1))
        
    state_code, county_code = args.state, args.county
    categories = resolve_categories(args.category)

    print(f"{'='*60}")
    print(f"EPA Data - County Aggregation")
    print(f"State: {state_code}, County: {county_code}, Years: {years}")
    print(f"Categories: {', '.join(categories)}")
    print(f"{'='*60}")

    county_name = None
    summary = {}

    for cat_key in categories:
        cat = CATEGORIES[cat_key]
        print(f"\n{'━'*60}")
        print(f"CATEGORY: {cat_key.upper()}")
        print(f"{'━'*60}")

        cat_results = {}
        for pc, pcfg in cat.get("params", {}).items():
            if pcfg.get("is_aqi"):
                r, cn = process_aqi(years, state_code, county_code)
            else:
                r, cn = process_one_param(cat["raw_subdir"], pc, pcfg,
                                          state_code, county_code, years)
            if r is not None:
                cat_results[pcfg["out_name"]] = r
            if cn and not county_name:
                county_name = cn

        if not cat_results:
            print(f"\n  No data for {cat_key}.")
            summary[cat_key] = "NO DATA"
            continue

        # Save
        dir_name = f"{state_code:02d}_{county_code:03d}_{county_name or 'unknown'}"
        out_dir = os.path.join(".", "data", "processed", dir_name, cat["out_subdir"])
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  Saving to: {out_dir}")
        files = []
        for pname, df in cat_results.items():
            fn = f"{pname}_daily.csv"
            df.to_csv(os.path.join(out_dir, fn), index=False)
            print(f"    {fn:40s} ({len(df):>5} rows)")
            files.append(fn)

        mn = cat.get("merge_name")
        if mn and len(cat_results) > 1:
            is_multi = any(p.get("is_multi_param") for p in cat["params"].values())
            m = merge_results(cat_results, is_multi)
            if m is not None:
                mfn = f"{mn}.csv"
                m.to_csv(os.path.join(out_dir, mfn), index=False)
                print(f"    {mfn:40s} ({len(m):>5} rows)")
                files.append(mfn)

        summary[cat_key] = files

    # Final summary
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    dn = f"{state_code:02d}_{county_code:03d}_{county_name or 'unknown'}"
    print(f"Output: {os.path.abspath(os.path.join('.', 'data', 'processed', dn))}\n")
    for ck, st in summary.items():
        if isinstance(st, list):
            print(f"  {ck}/ ({len(st)} files)")
            for f in st:
                print(f"    └── {f}")
        else:
            print(f"  {ck}/ -> {st}")


if __name__ == "__main__":
    main()
