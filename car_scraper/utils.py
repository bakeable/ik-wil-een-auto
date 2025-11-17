"""Utility functions for data cleaning and processing."""

import re
import pandas as pd
import argparse
from pathlib import Path
from typing import Optional


def normalize_brand_model(brand: str, model: str) -> str:
    """Normalize brand and model names for file naming.

    Args:
        brand: Car brand name (e.g., "Peugeot", "Volkswagen")
        model: Car model name (e.g., "2008", "Golf GTI")

    Returns:
        Normalized string like "peugeot.2008" or "volkswagen.golf_gti"
    """
    # Normalize brand: lowercase, remove spaces and special chars
    brand_clean = re.sub(r"[^\w]", "", brand.lower())

    # Normalize model: lowercase, spaces to underscores, remove special chars except underscores
    model_clean = re.sub(r"[^\w\s]", "", model.lower())
    model_clean = re.sub(r"\s+", "_", model_clean.strip())

    return f"{brand_clean}.{model_clean}"


def get_brand_model_paths(brand: str, model: str) -> dict:
    """Get standardized file paths for a specific brand/model.

    Args:
        brand: Car brand name
        model: Car model name

    Returns:
        Dictionary with standardized paths for data and reports
    """
    normalized = normalize_brand_model(brand, model)

    return {
        "data_file": f"data/{normalized}.listings.csv",
        "standard_report": f"reports/{normalized}.analysis_report.html",
        "standard_deals": f"reports/{normalized}.best_deals.csv",
        "personal_report": f"reports/{normalized}.personal_analysis_report.html",
        "personal_deals": f"reports/{normalized}.personal_best_deals.csv",
        "normalized_name": normalized,
    }


def clean_price(price_str: str) -> Optional[int]:
    """Clean and parse price string to integer."""
    if not price_str:
        return None

    # Remove currency symbols and spaces, keep only digits
    cleaned = re.sub(r"[^\d]", "", price_str)

    try:
        return int(cleaned)
    except ValueError:
        return None


def clean_mileage(km_str: str) -> Optional[int]:
    """Clean and parse mileage string to integer."""
    if not km_str:
        return None

    # Remove 'km' and other non-digit characters except dots and commas
    cleaned = re.sub(r"[^\d.,]", "", km_str)
    # Remove dots and commas used as thousands separators
    cleaned = cleaned.replace(".", "").replace(",", "")

    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_date(date_str: str) -> tuple[Optional[int], Optional[int]]:
    """Parse date string in MM/YYYY format."""
    if not date_str:
        return None, None

    match = re.match(r"(\d{2})/(\d{4})", date_str)
    if match:
        month = int(match.group(1))
        year = int(match.group(2))
        return month, year

    return None, None


def clean_horsepower(hp_str: str) -> Optional[int]:
    """Extract horsepower from string."""
    if not hp_str:
        return None

    # Look for number followed by 'hp' or 'HP'
    match = re.search(r"(\d+)\s*hp", hp_str, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def normalize_fuel_type(fuel_str: str) -> Optional[str]:
    """Normalize fuel type strings."""
    if not fuel_str:
        return None

    fuel_mapping = {
        "electric/gasoline": "Hybrid Electric/Gasoline",
        "hybrid (electric/gasoline)": "Hybrid Electric/Gasoline",
        "hybrid (electric/diesel)": "Hybrid Electric/Diesel",
        "gasoline": "Gasoline",
        "diesel": "Diesel",
        "electric": "Electric",
        "cng": "CNG",
        "lpg": "LPG",
        "hydrogen": "Hydrogen",
        "ethanol": "Ethanol",
    }

    normalized = fuel_str.lower().strip()
    return fuel_mapping.get(normalized, fuel_str.title())


def calculate_depreciation_rate(
    price: int, age_years: float, original_price: Optional[int] = None
) -> Optional[float]:
    """Calculate annual depreciation rate."""
    if original_price is None or age_years <= 0:
        return None

    # Simple depreciation rate calculation
    total_depreciation = (original_price - price) / original_price
    annual_rate = total_depreciation / age_years

    return annual_rate


def detect_outliers_iqr(values: list, multiplier: float = 1.5) -> list:
    """Detect outliers using the IQR method."""
    if len(values) < 4:
        return []

    sorted_values = sorted(values)
    n = len(sorted_values)

    q1_index = n // 4
    q3_index = 3 * n // 4

    q1 = sorted_values[q1_index]
    q3 = sorted_values[q3_index]
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return [val for val in values if val < lower_bound or val > upper_bound]


def clean_duplicates(
    input_file: str,
    output_file: Optional[str] = None,
    duplicate_columns: Optional[list] = None,
) -> None:
    """Remove duplicate car listings based on specified columns.

    Removes duplicates where mileage, age and horsepower are the same,
    keeping the newest row (last occurrence) and removing older ones.

    Args:
        input_file: Path to the CSV file to clean
        output_file: Path for cleaned output file (defaults to overwriting input)
        duplicate_columns: Columns to use for duplicate detection
                          (defaults to ['kilometers', 'age_in_years', 'horsepower'])
    """
    if duplicate_columns is None:
        duplicate_columns = ["kilometers", "age_in_years", "horsepower"]

    # Load the data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Initial record count: {initial_count}")

    # Check which columns exist in the data
    available_columns = [col for col in duplicate_columns if col in df.columns]
    missing_columns = [col for col in duplicate_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Columns not found in data: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")

    if not available_columns:
        print("Error: None of the duplicate detection columns found in data!")
        return

    print(f"Using columns for duplicate detection: {available_columns}")

    # Find duplicates
    duplicates_mask = df.duplicated(subset=available_columns, keep=False)
    duplicates_count = duplicates_mask.sum()

    if duplicates_count == 0:
        print("No duplicates found!")
        return

    print(f"Found {duplicates_count} rows with duplicates")

    # Show duplicate groups before cleaning
    duplicate_groups = df[duplicates_mask].groupby(available_columns).size()
    print(f"Number of duplicate groups: {len(duplicate_groups)}")
    print(f"Largest duplicate group size: {duplicate_groups.max()}")

    # Remove duplicates, keeping the last occurrence (newest)
    df_clean = df.drop_duplicates(subset=available_columns, keep="last")
    final_count = len(df_clean)
    removed_count = initial_count - final_count

    print(f"Final record count: {final_count}")
    print(f"Removed {removed_count} duplicate rows")

    # Save the cleaned data
    output_path = output_file if output_file else input_file
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

    # Show some statistics about what was removed
    if removed_count > 0:
        print("\n📊 Cleaning Summary:")
        print(f"   Removed: {removed_count} duplicate rows")
        print(f"   Kept:    {final_count} unique rows")
        print(f"   Savings: {removed_count/initial_count*100:.1f}% reduction")


def main():
    """Command-line interface for the duplicate cleaner."""
    parser = argparse.ArgumentParser(
        description="Clean duplicate car listings from CSV file"
    )
    parser.add_argument("brand", help="Car brand name (e.g., Peugeot, Volkswagen)")
    parser.add_argument("model", help="Car model name (e.g., 2008, Golf GTI)")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["kilometers", "age_in_years", "horsepower"],
        help="Columns to use for duplicate detection (default: kilometers age_in_years horsepower)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview duplicates without removing them",
    )

    args = parser.parse_args()

    # Get standardized file paths
    paths = get_brand_model_paths(args.brand, args.model)
    input_file = paths["data_file"]

    # Validate input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist")
        print(f"Expected file for {args.brand} {args.model}: {input_file}")
        return

    if args.preview:
        # Preview mode - just show what would be removed
        df = pd.read_csv(input_file)
        print(
            f"Loaded {len(df)} records for {args.brand} {args.model} from {input_file}"
        )

        available_columns = [col for col in args.columns if col in df.columns]
        if not available_columns:
            print(f"Error: None of the specified columns found: {args.columns}")
            print(f"Available columns: {list(df.columns)}")
            return

        duplicates_mask = df.duplicated(subset=available_columns, keep=False)
        duplicates = df[duplicates_mask]

        if len(duplicates) == 0:
            print("No duplicates found!")
            return

        print(f"\n📋 Found {len(duplicates)} duplicate rows:")
        duplicate_groups = duplicates.groupby(available_columns)

        for name, group in duplicate_groups:
            print(f"\n🔍 Duplicate group ({len(group)} rows):")
            print(f"   {dict(zip(available_columns, name))}")
            if len(group) <= 5:
                for idx, row in group.iterrows():
                    print(f"   Row {idx}: Price={row.get('price', 'N/A')}")
            else:
                print(f"   ... {len(group)} rows total")
    else:
        # Actually clean the data
        print(f"Cleaning duplicates for {args.brand} {args.model}...")
        clean_duplicates(
            input_file, input_file, args.columns
        )  # Always overwrite source


if __name__ == "__main__":
    main()
