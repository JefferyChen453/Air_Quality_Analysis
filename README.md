# Air Quality Analysis

## Dataset

- **Data source:** [EPA Air Quality Data (AQS)](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
- **Content:** Daily summary CSVs by pollutant (criteria gases, particulates, meteorological, toxics/precursors/lead, daily AQI by county).
- **Use case:** Environmental health tracking, pollution level and trend analysis, and visualization.

For simplicity, we only analyze Los Angeles County (State - County Code: 6 - 37) in Year 2025.


## Setup

```bash
uv sync
```

## Pipeline

### 1. Download raw data

For simplicity, hourly data is NOT downloaded or used. Only daily summary data is downloaded and unzipped into `./data/raw/daily_summary/`.

| Category            | File Contents                                      |
|---------------------|-----------------------------------------------|
| Criteria Gases      | O₃ (44201), SO₂ (42401), CO (42101), NO₂ (42602) |
| Particulates        | PM2.5 FRM (88101), non-FRM (88502), PM10 (81102), PMc (86101) |
| Meteorological      | Wind, Temperature, Pressure, RH/Dewpoint      |
| Toxics/Precursors   | HAPS, VOCS, NONOxNOy, Lead                    |
| Daily AQI           | County-level AQI                              |

```bash
uv run data_process/download_files.py --year 2025
# all categories above of year 2025
```

**Data directory tree after downloading:**

```
data/raw/daily_summary/
├── criteria_gases/       # daily_44201_2025.csv, ...
├── particulates/
├── meteorological/
├── toxics_precursors_lead/
└── daily_aqi/            # daily_aqi_by_county_2025.csv
```

---

### 2. Clean and aggregate by county
In raw daily data, each county contains multiple entries from multiple sites, and each site may have multiple observations (i.e., multiple rows for one day). To aggregate them into 1 entry per day, we aggregate across sites and POC by averaging and deduplication.

Specify state code and county code (e.g. LA County: 6 - 37), the cleaning&aggregation pipeline involves:
 - cleaning records (event type, sample duration, pollutant standard, POC dedup)
 - aggregating to **county-level daily** series.

Commands for data cleaning and aggregation for LA County:
```bash
uv run data_process/clean_and_aggregate.py --state 6 --county 37
```

If no state or county is specified, the script automatically detects all available counties from the raw dataset and processes them sequentially.

Commands for data cleaning and aggregation if both state and county are omitted:
```bash
uv run data_process/clean_and_aggregate.py
```

**Processed layout for a single county (aligned with original raw data)**

```
data/processed/06_037_Los_Angeles/
├── criteria_gases/       # ozone_daily.csv, all_criteria_gases_daily.csv, ...
├── particulates/
├── meteorological/
├── toxics_precursors_lead/
└── daily_aqi/            # aqi_daily.csv
```

### 3. Visualize

Reads from `data/processed/{state}_{county}_{name}/` and writes figures to `figures/{state}_{county}_{name}/`.

```bash
uv run data_process/visualize.py --state 6 --county 37 # for LA County
```

**Generated figures:**

| #  | File                              | Description |
|----|-----------------------------------|-------------|
| 01 | `01_aqi_timeseries.png`           | Daily AQI over time, color-coded by category |
| 02 | `02_aqi_distribution.png`         | AQI histogram, category pie, defining pollutant |
| 03 | `03_aqi_monthly_box.png`          | Monthly AQI box plots |
| 04 | `04_criteria_gases_timeseries.png`| O₃, NO₂, CO, SO₂ daily trends + 7-day MA |
| 05 | `05_gases_monthly_box.png`        | Monthly box plots for each criteria gas |
| 06 | `06_particulates_timeseries.png` | PM2.5 and PM10 daily trends, NAAQS reference |
| 07 | `07_pm_monthly_box.png`           | Monthly PM2.5 and PM10 box plots |
| 08 | `08_meteorological_timeseries.png` | Temp, humidity, wind, pressure |
| 09 | `09_pollutant_weather_scatter.png`| Pollutant vs temperature/RH/wind with regression |
| 10 | `10_aqi_vs_pollutants.png`       | AQI vs O₃, NO₂, PM2.5 by AQI category |

MA refers to Moving Average

---

### 4. Train CatBoost models

The `models/train_catboost.py` script merges pollutant, particulate, meteorological, and AQI tables, engineers lag/rolling features, and trains a CatBoost model with time-based splits (train = years before validation year, validation = `--val-year`, test = `--test-year`).

```bash
# Numeric AQI regression (defaults to state=6, county=37, val-year=2023, test-year=2024)
uv run python models/train_catboost.py --model-type regression

# AQI category classification
uv run python models/train_catboost.py --model-type classification
```

Artifacts (model binaries + metrics JSON) are written to `models/{state}_{county}_{name}/`. Use `--output-dir` to override the destination, and `--val-year` / `--test-year` to change the temporal splits.

---
