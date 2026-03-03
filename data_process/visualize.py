"""
Visualization for Los Angeles County (06-037), Year 2025.

Generates figures:
  - AQI overview (time-series, distribution, category breakdown, defining pollutant)
  - Criteria-gas trends (O3, SO2, CO, NO2)
  - Particulate matter trends (PM2.5, PM10)
  - Meteorological context (temperature, humidity, wind, pressure)
  - Monthly box-plots for key pollutants
  - Pollutant–weather scatter matrix
  - AQI vs key pollutants

Usage:
    uv run data_process/visualize.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Config
DATA_DIR = os.path.join("data", "processed", "06_037_Los_Angeles")
FIG_DIR = os.path.join("figures")
os.makedirs(FIG_DIR, exist_ok=True)

AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}
AQI_ORDER = list(AQI_COLORS.keys())

PALETTE = sns.color_palette("tab10")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# Data loading
def load():
    data = {}

    aqi_path = os.path.join(DATA_DIR, "daily_aqi", "aqi_daily.csv")
    aqi = pd.read_csv(aqi_path)
    aqi["Date"] = pd.to_datetime(aqi["Date"])
    aqi["month"] = aqi["Date"].dt.month
    aqi["dow"] = aqi["Date"].dt.dayofweek
    data["aqi"] = aqi

    gases_path = os.path.join(DATA_DIR, "criteria_gases", "all_criteria_gases_daily.csv")
    gases = pd.read_csv(gases_path)
    gases["Date Local"] = pd.to_datetime(gases["Date Local"])
    gases["month"] = gases["Date Local"].dt.month
    data["gases"] = gases

    for gas in ["ozone", "so2", "co", "no2"]:
        p = os.path.join(DATA_DIR, "criteria_gases", f"{gas}_daily.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["Date Local"] = pd.to_datetime(df["Date Local"])
            df["month"] = df["Date Local"].dt.month
            data[gas] = df

    pm_path = os.path.join(DATA_DIR, "particulates", "all_particulates_daily.csv")
    pm = pd.read_csv(pm_path)
    pm["Date Local"] = pd.to_datetime(pm["Date Local"])
    pm["month"] = pm["Date Local"].dt.month
    data["pm"] = pm

    for ptype in ["pm25_frm", "pm25_nonfrm", "pm10"]:
        p = os.path.join(DATA_DIR, "particulates", f"{ptype}_daily.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["Date Local"] = pd.to_datetime(df["Date Local"])
            df["month"] = df["Date Local"].dt.month
            data[ptype] = df

    met_path = os.path.join(DATA_DIR, "meteorological", "all_meteorological_daily.csv")
    met = pd.read_csv(met_path)
    met["Date Local"] = pd.to_datetime(met["Date Local"])
    met["month"] = met["Date Local"].dt.month
    data["met"] = met

    print(f"Loaded {len(data)} datasets from {DATA_DIR}")
    return data


# 1. AQI
def plot_aqi_timeseries(aqi):
    fig, ax = plt.subplots(figsize=(14, 4.5))

    colors = aqi["Category"].map(AQI_COLORS).fillna("#999999")
    ax.bar(aqi["Date"], aqi["AQI"], color=colors, width=1.0, edgecolor="none")

    thresholds = [(50, "Good"), (100, "Moderate"), (150, "USG"), (200, "Unhealthy")]
    for val, label in thresholds:
        ax.axhline(val, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)

    ax.set_title("Daily AQI – Los Angeles County, 2025")
    ax.set_ylabel("AQI")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(aqi["Date"].min() - pd.Timedelta(days=2),
                aqi["Date"].max() + pd.Timedelta(days=2))
    ax.set_ylim(0, max(aqi["AQI"].max() * 1.1, 160))

    from matplotlib.patches import Patch
    present = aqi["Category"].unique()
    legend_handles = [Patch(facecolor=AQI_COLORS.get(c, "#999"), label=c)
                      for c in AQI_ORDER if c in present]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    savefig(fig, "01_aqi_timeseries.png")


def plot_aqi_distribution(aqi):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # histogram
    ax = axes[0]
    ax.hist(aqi["AQI"], bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(aqi["AQI"].median(), color="red", linestyle="--", label=f'Median={aqi["AQI"].median():.0f}')
    ax.axvline(aqi["AQI"].mean(), color="orange", linestyle="--", label=f'Mean={aqi["AQI"].mean():.1f}')
    ax.set_xlabel("AQI")
    ax.set_ylabel("Days")
    ax.set_title("AQI Distribution")
    ax.legend(fontsize=8)

    # category pie
    ax = axes[1]
    counts = aqi["Category"].value_counts()
    ordered = [c for c in AQI_ORDER if c in counts.index]
    vals = [counts[c] for c in ordered]
    clrs = [AQI_COLORS[c] for c in ordered]
    short = [c.replace("Unhealthy for Sensitive Groups", "USG") for c in ordered]
    wedges, texts, autotexts = ax.pie(
        vals, labels=short, colors=clrs, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 8})
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("AQI Category Breakdown")

    # defining parameter
    ax = axes[2]
    dp = aqi["Defining Parameter"].value_counts()
    ax.barh(dp.index, dp.values, color=PALETTE[1:len(dp)+1])
    ax.set_xlabel("Days")
    ax.set_title("Defining Pollutant Frequency")
    ax.invert_yaxis()

    fig.suptitle("AQI Statistics – LA County 2025", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "02_aqi_distribution.png")


def plot_aqi_monthly(aqi):
    fig, ax = plt.subplots(figsize=(10, 5))

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    aqi["month_name"] = aqi["month"].map(month_names)
    order = [month_names[m] for m in sorted(aqi["month"].unique())]

    sns.boxplot(data=aqi, x="month_name", y="AQI", order=order,
                palette="YlOrRd", ax=ax, fliersize=3)
    ax.axhline(50, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="Good/Moderate")
    ax.axhline(100, color="orange", linewidth=0.8, linestyle="--", alpha=0.5, label="Moderate/USG")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI")
    ax.set_title("Monthly AQI Distribution – LA County 2025")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "03_aqi_monthly_box.png")


# 2. Criteria gases
def plot_criteria_gases_ts(gases):
    pollutants = [
        ("ozone_daily_mean_conc", "ozone_daily_peak_conc", "Ozone (O₃)", "ppm"),
        ("no2_daily_mean_conc", "no2_daily_peak_conc", "NO₂", "ppb"),
        ("co_daily_mean_conc", "co_daily_peak_conc", "CO", "ppm"),
        ("so2_daily_mean_conc", "so2_daily_peak_conc", "SO₂", "ppb"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for ax, (mean_col, peak_col, title, unit) in zip(axes, pollutants):
        if mean_col not in gases.columns:
            continue
        dates = gases["Date Local"]
        ax.plot(dates, gases[mean_col], color=PALETTE[0], linewidth=0.8, label="Daily Mean")
        ax.fill_between(dates, gases[mean_col], alpha=0.15, color=PALETTE[0])
        if peak_col in gases.columns:
            ax.plot(dates, gases[peak_col], color=PALETTE[3], linewidth=0.6,
                    alpha=0.7, label="Daily Peak")

        roll = gases[mean_col].rolling(7, center=True, min_periods=3).mean()
        ax.plot(dates, roll, color=PALETTE[1], linewidth=1.8, label="7-day MA")

        ax.set_ylabel(f"{title} ({unit})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_title("Criteria Gases – Daily Trends, LA County 2025")
    fig.tight_layout()
    savefig(fig, "04_criteria_gases_timeseries.png")


def plot_gases_monthly_box(data):
    gases_list = [
        ("ozone", "daily_mean_conc", "Ozone (O₃) ppm"),
        ("no2", "daily_mean_conc", "NO₂ (ppb)"),
        ("co", "daily_mean_conc", "CO (ppm)"),
        ("so2", "daily_mean_conc", "SO₂ (ppb)"),
    ]
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (key, col, ylabel) in zip(axes.flat, gases_list):
        if key not in data:
            continue
        df = data[key].copy()
        df["month_name"] = df["month"].map(month_names)
        order = [month_names[m] for m in sorted(df["month"].unique())]
        sns.boxplot(data=df, x="month_name", y=col, order=order,
                    palette="Blues_d", ax=ax, fliersize=3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Monthly Criteria Gas Distributions – LA County 2025", fontsize=14)
    fig.tight_layout()
    savefig(fig, "05_gases_monthly_box.png")


# 3. Particulates
def plot_particulates_ts(pm, data):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # PM2.5
    ax = axes[0]
    if "pm25_frm" in data:
        df = data["pm25_frm"]
        ax.plot(df["Date Local"], df["daily_mean_conc"], color=PALETTE[0],
                linewidth=0.8, label="PM2.5 FRM Mean")
        ax.fill_between(df["Date Local"], df["daily_mean_conc"], alpha=0.12, color=PALETTE[0])
        roll = df["daily_mean_conc"].rolling(7, center=True, min_periods=3).mean()
        ax.plot(df["Date Local"], roll, color=PALETTE[1], linewidth=1.8, label="7-day MA")

    if "pm25_nonfrm_daily_mean_conc" in pm.columns:
        ax.plot(pm["Date Local"], pm["pm25_nonfrm_daily_mean_conc"],
                color=PALETTE[2], linewidth=0.6, alpha=0.7, label="PM2.5 non-FRM Mean")

    ax.axhline(35, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="NAAQS 24-hr (35 µg/m³)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # PM10
    ax = axes[1]
    if "pm10_daily_mean_conc" in pm.columns:
        s = pm["pm10_daily_mean_conc"].dropna()
        dates_pm10 = pm.loc[s.index, "Date Local"]
        ax.plot(dates_pm10, s, color=PALETTE[4], linewidth=0.8, label="PM10 Mean")
        ax.fill_between(dates_pm10, s, alpha=0.12, color=PALETTE[4])
        roll10 = s.rolling(7, center=True, min_periods=3).mean()
        ax.plot(dates_pm10, roll10, color=PALETTE[5], linewidth=1.8, label="7-day MA")
    ax.axhline(150, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="NAAQS 24-hr (150 µg/m³)")
    ax.set_ylabel("PM10 (µg/m³)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_title("Particulate Matter – Daily Trends, LA County 2025")
    fig.tight_layout()
    savefig(fig, "06_particulates_timeseries.png")


def plot_pm_monthly_box(data):
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if "pm25_frm" in data:
        df = data["pm25_frm"].copy()
        df["month_name"] = df["month"].map(month_names)
        order = [month_names[m] for m in sorted(df["month"].unique())]
        sns.boxplot(data=df, x="month_name", y="daily_mean_conc", order=order,
                    palette="Oranges", ax=axes[0], fliersize=3)
        axes[0].axhline(35, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        axes[0].set_ylabel("PM2.5 FRM (µg/m³)")
        axes[0].set_xlabel("")
        axes[0].set_title("PM2.5 by Month")
        axes[0].tick_params(axis="x", rotation=45)

    if "pm10" in data:
        df = data["pm10"].copy()
        df["month_name"] = df["month"].map(month_names)
        order = [month_names[m] for m in sorted(df["month"].unique())]
        sns.boxplot(data=df, x="month_name", y="daily_mean_conc", order=order,
                    palette="Reds", ax=axes[1], fliersize=3)
        axes[1].set_ylabel("PM10 (µg/m³)")
        axes[1].set_xlabel("")
        axes[1].set_title("PM10 by Month")
        axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Monthly Particulate Distributions – LA County 2025", fontsize=14)
    fig.tight_layout()
    savefig(fig, "07_pm_monthly_box.png")


# 4. Meteorological
def plot_meteorological(met):
    met_vars = [
        ("outdoor_temperature_mean", "outdoor_temperature_peak", "Temperature (°F)", "Reds_r"),
        ("relative_humidity__mean", "relative_humidity__peak", "Relative Humidity (%)", "Blues"),
        ("wind_speed___resultant_mean", "wind_speed___resultant_peak", "Wind Speed (knots)", "Greens"),
        ("barometric_pressure_mean", "barometric_pressure_peak", "Pressure (mbar)", "Purples"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for ax, (mean_col, peak_col, ylabel, cmap) in zip(axes, met_vars):
        if mean_col not in met.columns:
            ax.set_visible(False)
            continue

        dates = met["Date Local"]
        vals = met[mean_col].dropna()
        ax.plot(dates, met[mean_col], color=sns.color_palette(cmap, 5)[2],
                linewidth=0.8, label="Daily Mean")
        if peak_col in met.columns:
            ax.plot(dates, met[peak_col], color=sns.color_palette(cmap, 5)[4],
                    linewidth=0.5, alpha=0.5, label="Daily Peak")
        roll = met[mean_col].rolling(14, center=True, min_periods=5).mean()
        ax.plot(dates, roll, color=sns.color_palette(cmap, 5)[3],
                linewidth=2, label="14-day MA")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_title("Meteorological Conditions – LA County 2025")
    fig.tight_layout()
    savefig(fig, "08_meteorological_timeseries.png")


# 5. pollutant vs weather
def _build_combined(data):
    """Build combined DataFrame of pollutants + weather for scatter/correlation use."""
    gases = data["gases"]
    pm = data["pm"]
    met = data["met"]
    cols = {
        "O₃ mean": gases.set_index("Date Local")["ozone_daily_mean_conc"],
        "NO₂ mean": gases.set_index("Date Local")["no2_daily_mean_conc"],
        "CO mean": gases.set_index("Date Local")["co_daily_mean_conc"],
        "SO₂ mean": gases.set_index("Date Local")["so2_daily_mean_conc"],
    }
    if "pm25_frm_daily_mean_conc" in pm.columns:
        cols["PM2.5 mean"] = pm.set_index("Date Local")["pm25_frm_daily_mean_conc"]
    if "pm10_daily_mean_conc" in pm.columns:
        cols["PM10 mean"] = pm.set_index("Date Local")["pm10_daily_mean_conc"]
    if "outdoor_temperature_mean" in met.columns:
        cols["Temp (°F)"] = met.set_index("Date Local")["outdoor_temperature_mean"]
    if "relative_humidity__mean" in met.columns:
        cols["RH (%)"] = met.set_index("Date Local")["relative_humidity__mean"]
    if "wind_speed___resultant_mean" in met.columns:
        cols["Wind (kn)"] = met.set_index("Date Local")["wind_speed___resultant_mean"]
    return pd.DataFrame(cols)


def plot_pollutant_weather_scatter(data):
    combined = _build_combined(data)
    weather = ["Temp (°F)", "RH (%)", "Wind (kn)"]
    pollutants = ["O₃ mean", "NO₂ mean", "PM2.5 mean"]
    pollutants = [p for p in pollutants if p in combined.columns]
    weather = [w for w in weather if w in combined.columns]

    if not pollutants or not weather:
        return

    nrow, ncol = len(pollutants), len(weather)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow))
    if nrow == 1:
        axes = axes[np.newaxis, :]
    if ncol == 1:
        axes = axes[:, np.newaxis]

    for i, poll in enumerate(pollutants):
        for j, wvar in enumerate(weather):
            ax = axes[i, j]
            sub = combined[[poll, wvar]].dropna()
            if len(sub) < 5:
                ax.set_visible(False)
                continue

            ax.scatter(sub[wvar], sub[poll], s=12, alpha=0.5, color=PALETTE[i])

            slope, intercept, r, p_val, _ = stats.linregress(sub[wvar], sub[poll])
            x_fit = np.linspace(sub[wvar].min(), sub[wvar].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.2)
            ax.annotate(f"r={r:.2f}\np={p_val:.1e}", xy=(0.05, 0.92),
                        xycoords="axes fraction", fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            ax.set_xlabel(wvar)
            ax.set_ylabel(poll)
            ax.grid(alpha=0.2)

    fig.suptitle("Pollutant vs Weather – LA County 2025", fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, "09_pollutant_weather_scatter.png")


# 6. AQI vs pollutant
def plot_aqi_vs_pollutants(data):
    aqi = data["aqi"]
    gases = data["gases"]
    pm = data["pm"]

    merged = aqi.merge(gases, left_on="Date", right_on="Date Local", how="inner")
    merged = merged.merge(pm, left_on="Date", right_on="Date Local", how="left",
                          suffixes=("", "_pm"))

    pairs = [
        ("ozone_daily_mean_conc", "O₃ (ppm)"),
        ("no2_daily_mean_conc", "NO₂ (ppb)"),
        ("pm25_frm_daily_mean_conc", "PM2.5 (µg/m³)"),
    ]
    pairs = [(c, l) for c, l in pairs if c in merged.columns]

    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, pairs):
        sub = merged[["AQI", col, "Category"]].dropna()
        for cat in AQI_ORDER:
            mask = sub["Category"] == cat
            if mask.sum() == 0:
                continue
            ax.scatter(sub.loc[mask, col], sub.loc[mask, "AQI"],
                       s=18, alpha=0.6, color=AQI_COLORS.get(cat, "#999"),
                       label=cat.replace("Unhealthy for Sensitive Groups", "USG"))

        if len(sub) > 5:
            slope, intercept, r, _, _ = stats.linregress(sub[col], sub["AQI"])
            x_fit = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color="black",
                    linewidth=1.2, linestyle="--")
            ax.set_title(f"AQI vs {label}  (r={r:.2f})")
        else:
            ax.set_title(f"AQI vs {label}")

        ax.set_xlabel(label)
        ax.set_ylabel("AQI")
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(AQI_ORDER),
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("AQI vs Key Pollutants – LA County 2025", fontsize=14)
    fig.tight_layout()
    savefig(fig, "10_aqi_vs_pollutants.png")


# 7. Summary
def print_summary_stats(data):
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS – LA County 2025")
    print("=" * 80)

    aqi = data["aqi"]
    print(f"\nAQI:")
    print(f"  Days recorded:  {len(aqi)}")
    print(f"  Mean:           {aqi['AQI'].mean():.1f}")
    print(f"  Median:         {aqi['AQI'].median():.0f}")
    print(f"  Max:            {aqi['AQI'].max()}")
    print(f"  Min:            {aqi['AQI'].min()}")
    print(f"  Std:            {aqi['AQI'].std():.1f}")
    print(f"  Category counts:")
    for cat in AQI_ORDER:
        n = (aqi["Category"] == cat).sum()
        if n > 0:
            print(f"    {cat:40s} {n:>4d} days ({n/len(aqi)*100:.1f}%)")
    print(f"  Top defining pollutants:")
    for poll, cnt in aqi["Defining Parameter"].value_counts().items():
        print(f"    {poll:40s} {cnt:>4d} days ({cnt/len(aqi)*100:.1f}%)")

    for key, label, unit in [("ozone", "Ozone", "ppm"), ("no2", "NO₂", "ppb"),
                              ("co", "CO", "ppm"), ("so2", "SO₂", "ppb"),
                              ("pm25_frm", "PM2.5 FRM", "µg/m³"), ("pm10", "PM10", "µg/m³")]:
        if key in data:
            df = data[key]
            col = "daily_mean_conc"
            if col in df.columns:
                s = df[col].dropna()
                print(f"\n{label} ({unit}):")
                print(f"  Days: {len(s)}, Mean: {s.mean():.4f}, "
                      f"Median: {s.median():.4f}, Max: {s.max():.4f}, "
                      f"Std: {s.std():.4f}")


# Main
def main():
    data = load()

    print("\n Visualizing")

    plot_aqi_timeseries(data["aqi"])

    plot_aqi_distribution(data["aqi"])

    plot_aqi_monthly(data["aqi"])

    plot_criteria_gases_ts(data["gases"])

    plot_gases_monthly_box(data)

    plot_particulates_ts(data["pm"], data)

    plot_pm_monthly_box(data)

    plot_meteorological(data["met"])

    plot_pollutant_weather_scatter(data)

    plot_aqi_vs_pollutants(data)

    print_summary_stats(data)

    print(f"\nAll figures saved to: {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
