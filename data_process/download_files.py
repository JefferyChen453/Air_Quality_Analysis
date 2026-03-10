import os
import zipfile
import urllib.request
import sys
import argparse

# Config
BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"
CATEGORIES = {
    "criteria_gases": {
        "subdir": "criteria_gases",
        "files": {
            "44201": "Ozone (O3)",
            "42401": "SO2",
            "42101": "CO",
            "42602": "NO2",
        },
        "prefix": "daily",
    },
    "particulates": {
        "subdir": "particulates",
        "files": {
            "88101": "PM2.5 FRM/FEM Mass",
            "88502": "PM2.5 non-FRM/FEM Mass",
            "81102": "PM10 Mass",
            "86101": "PMc Mass",
        },
        "prefix": "daily",
    },
    "meteorological": {
        "subdir": "meteorological",
        "files": {
            "WIND": "Wind (Resultant)",
            "TEMP": "Temperature",
            "PRESS": "Barometric Pressure",
            "RH_DP": "RH and Dewpoint",
        },
        "prefix": "daily",
    },
    "toxics": {
        "subdir": "toxics_precursors_lead",
        "files": {
            "HAPS": "Hazardous Air Pollutants",
            "VOCS": "Volatile Organic Compounds",
            "NONOxNOy": "NO, NOx, NOy (Ozone Precursors)",
            "LEAD": "Lead",
        },
        "prefix": "daily",
    },
    "aqi": {
        "subdir": "daily_aqi",
        "files": {
            "aqi_by_county": "Daily AQI by County",
        },
        "prefix": "daily",
    },
}

CATEGORY_ALIASES = {
    "gases": "criteria_gases",
    "gas": "criteria_gases",
    "pm": "particulates",
    "met": "meteorological",
    "meteo": "meteorological",
    "toxic": "toxics",
    "all": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download EPA AirData files.")
    parser.add_argument("--year", type=int, nargs="+", default=[2025], help="Data year(s) (e.g., 2025, or 2015 2025 for a range)")
    parser.add_argument("--category", nargs="*", default=None,
                        help="Categories to download. Options: "
                             "criteria_gases (or gases), particulates (or pm), "
                             "meteorological (or met), toxics, aqi, all. "
                             "Default: all")
    return parser.parse_args()


def resolve_categories(category_args):
    """
    Resolve category arguments (including aliases) to actual category keys.
    """
    if category_args is None or "all" in category_args:
        return list(CATEGORIES.keys())

    resolved = []
    for cat in category_args:
        cat_lower = cat.lower()
        if cat_lower in CATEGORIES:
            resolved.append(cat_lower)
        elif cat_lower in CATEGORY_ALIASES:
            alias = CATEGORY_ALIASES[cat_lower]
            if alias is None:  # for "all"
                return list(CATEGORIES.keys())
            resolved.append(alias)
        else:
            print(f"WARNING: Unknown category '{cat}', skipping.")
    return list(dict.fromkeys(resolved))


def download_file(url, dest_path):
    print(f"    URL:  {url}")
    print(f"    Save: {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n    ERROR: {e}")
        return False


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    Progress: {pct:5.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
    else:
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r    Downloaded: {mb:.1f} MB")
    sys.stdout.flush()


def extract_zip(zip_path, extract_dir):
    print(f"    Extracting: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
        extracted = zf.namelist()
        for name in extracted:
            print(f"      -> {name}")
    return extracted


def download_category(cat_key, cat_config, year, base_raw_dir):
    """Download all files for one category."""
    subdir = cat_config["subdir"]
    out_dir = os.path.join(base_raw_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    prefix = cat_config["prefix"]

    for file_key, description in cat_config["files"].items():
        zip_name = f"{prefix}_{file_key}_{year}.zip"
        url = f"{BASE_URL}/{zip_name}"
        zip_path = os.path.join(out_dir, zip_name)

        print(f"\n  [{description}] {zip_name}")

        success = download_file(url, zip_path)
        if not success:
            results[file_key] = "FAILED"
            continue

        try:
            extracted = extract_zip(zip_path, out_dir)
            results[file_key] = extracted
            os.remove(zip_path)
        except zipfile.BadZipFile:
            print(f"    ERROR: Bad zip file")
            results[file_key] = "BAD_ZIP"

    return results


def main():
    args = parse_args()
    years = args.year
    categories = resolve_categories(args.category)
    if len(years) == 2:
        years = list(range(years[0], years[1] + 1))
    
    base_raw_dir = os.path.join(".", "data", "raw", "daily_summary")

    print(f"{'='*60}")
    print(f"EPA AirData Downloader")
    print(f"Years: {years}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Output: {os.path.abspath(base_raw_dir)}")
    print(f"{'='*60}")

    all_results = {}

    for year in years:
        print(f"\n{'*'*60}")
        print(f"Processing Year: {year}")
        print(f"{'*'*60}")
        
        for cat_key in categories:
            cat_config = CATEGORIES[cat_key]
            print(f"\n{'━'*60}")
            print(f"Category: {cat_key.upper()}")
            print(f"{'━'*60}")

            results = download_category(cat_key, cat_config, year, base_raw_dir)
            if cat_key not in all_results:
                all_results[cat_key] = {}
            all_results[cat_key][year] = results

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for cat_key, year_results in all_results.items():
        print(f"\n  {cat_key}:")
        for year, results in year_results.items():
            print(f"    Year: {year}")
            for file_key, status in results.items():
                if isinstance(status, list):
                    print(f"      {file_key:20s}: OK -> {', '.join(status)}")
                else:
                    print(f"      {file_key:20s}: {status}")

    print(f"\nDirectory structure:")
    for cat_key in categories:
        subdir = CATEGORIES[cat_key]["subdir"]
        dir_path = os.path.join(base_raw_dir, subdir)
        if os.path.exists(dir_path):
            files = sorted(os.listdir(dir_path))
            print(f"\n  {dir_path}/")
            for f in files:
                size_mb = os.path.getsize(os.path.join(dir_path, f)) / (1024 * 1024)
                print(f"    {f:45s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
