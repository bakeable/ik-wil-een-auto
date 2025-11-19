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


def discover_available_models() -> list:
    """Discover all available car models by scanning the data folder.

    Returns:
        List of tuples (brand, model, normalized_name) for available models
    """
    data_path = Path("data")
    models = []

    for file_path in data_path.glob("*.listings.csv"):
        # Parse filename: brand.model.listings.csv
        filename = file_path.stem  # removes .csv
        if filename.endswith(".listings"):
            normalized_name = filename[:-9]  # remove '.listings'
            parts = normalized_name.split(".")
            if len(parts) >= 2:
                brand_part = parts[0]
                model_part = ".".join(parts[1:])  # handle models with dots

                # Convert back to readable format
                brand = brand_part.capitalize()
                model = model_part.replace("_", " ").title()

                models.append((brand, model, normalized_name, str(file_path)))

    return sorted(models)


def generate_analysis_index(
    models_data: list, output_path: str = "reports/index.html"
) -> None:
    """Generate an index.html file for easy navigation between car models.

    Args:
        models_data: List of (brand, model, normalized_name, analysis_data) tuples
        output_path: Path to save the index file
    """
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Car Analysis Dashboard</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background-color: #f5f5f5;
            }}
            h1, h2 {{ color: #333; }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .model-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); 
                gap: 20px; 
                margin: 30px 0;
            }}
            .model-card {{ 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 20px; 
                background: #fafafa;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .model-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .model-title {{ 
                font-size: 1.3em; 
                font-weight: bold; 
                margin-bottom: 15px; 
                color: #2c3e50;
            }}
            .stats {{ 
                background: white;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .stat-row {{ 
                display: flex; 
                justify-content: space-between; 
                margin: 5px 0;
                padding: 2px 0;
            }}
            .stat-label {{ font-weight: bold; color: #7f8c8d; }}
            .stat-value {{ color: #2c3e50; }}
            .links {{ margin-top: 15px; }}
            .btn {{ 
                display: inline-block;
                padding: 8px 16px; 
                margin: 5px 10px 5px 0; 
                text-decoration: none; 
                border-radius: 5px; 
                font-size: 0.9em;
                transition: background-color 0.3s;
            }}
            .btn-personal {{ 
                background: #3498db; 
                color: white; 
            }}
            .btn-personal:hover {{ background: #2980b9; }}
            .btn-standard {{ 
                background: #95a5a6; 
                color: white; 
            }}
            .btn-standard:hover {{ background: #7f8c8d; }}
            .summary {{ 
                background: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .highlight {{ 
                background: #d5dbdb;
                padding: 15px;
                border-left: 4px solid #3498db;
                margin: 15px 0;
            }}
            .no-data {{ 
                color: #e74c3c; 
                font-style: italic; 
            }}
            .last-updated {{
                color: #7f8c8d;
                font-size: 0.9em;
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Car Analysis Dashboard</h1>
            
            <div class="summary">
                <h2>Overview</h2>
                <p><strong>Available Models:</strong> {len(models_data)}</p>
                <p><strong>Total Listings Analyzed:</strong> {sum(data.get('total_listings', 0) for _, _, _, data in models_data)}</p>
            </div>
            
            <div class="highlight">
                <h3>💡 How to Use</h3>
                <ul>
                    <li><strong>Personal Analysis:</strong> Shows cost-effectiveness for your specific driving pattern</li>
                    <li><strong>Market Analysis:</strong> Shows general market value analysis and best deals</li>
                    <li><strong>Best Deal Cars:</strong> Optimized for your ownership period and annual mileage</li>
                </ul>
            </div>
            
            <h2>Car Models Analysis</h2>
            <div class="model-grid">
    """

    for brand, model, normalized_name, analysis_data in models_data:
        # Extract analysis data
        total_listings = analysis_data.get("total_listings", "N/A")
        best_deal_price = analysis_data.get("best_deal_price", "N/A")
        best_deal_cost_per_km = analysis_data.get("best_deal_cost_per_km", "N/A")
        avg_annual_cost = analysis_data.get("avg_annual_cost", "N/A")
        has_personal_analysis = analysis_data.get("has_personal_analysis", False)

        html_content += f"""
                <div class="model-card">
                    <div class="model-title">{brand} {model}</div>
                    <div class="stats">
                        <div class="stat-row">
                            <span class="stat-label">Total Listings:</span>
                            <span class="stat-value">{total_listings}</span>
                        </div>
        """

        if has_personal_analysis:
            html_content += f"""
                        <div class="stat-row">
                            <span class="stat-label">Best Deal Price:</span>
                            <span class="stat-value">€{best_deal_price:,.0f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Cost per KM:</span>
                            <span class="stat-value">€{best_deal_cost_per_km:.3f}/km</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Avg Annual Cost:</span>
                            <span class="stat-value">€{avg_annual_cost:,.0f}/year</span>
                        </div>
            """
        else:
            html_content += f"""
                        <div class="stat-row">
                            <span class="stat-label">Analysis Status:</span>
                            <span class="stat-value no-data">No personal analysis yet</span>
                        </div>
            """

        html_content += f"""
                    </div>
                    <div class="links">
        """

        if has_personal_analysis:
            html_content += f"""
                        <a href="{normalized_name}.personal_analysis_report.html" class="btn btn-personal">
                            📊 Personal Analysis
                        </a>
            """

        # Check if standard analysis exists
        standard_report_path = Path(f"reports/{normalized_name}.analysis_report.html")
        if standard_report_path.exists():
            html_content += f"""
                        <a href="{normalized_name}.analysis_report.html" class="btn btn-standard">
                            📈 Market Analysis
                        </a>
            """

        html_content += """
                    </div>
                </div>
        """

    html_content += f"""
            </div>
            
            <div class="last-updated">
                Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """

    # Write the file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✓ Analysis index generated: {output_path}")


def run_comprehensive_analysis(
    years_owned: int = 5, km_per_year: int = 30000, **filter_kwargs
):
    """Run personal analysis for all available car models and generate index.

    Args:
        years_owned: Number of years you plan to own the car
        km_per_year: Kilometers driven per year
        **filter_kwargs: Additional filters (max_age, max_price, etc.)
    """
    from .personal_analysis import PersonalCarAnalyzer

    print(f"🚗 Running comprehensive analysis for all models...")
    print(f"   Ownership period: {years_owned} years")
    print(f"   Annual driving: {km_per_year:,} km/year")
    print()

    # Discover available models
    available_models = discover_available_models()
    if not available_models:
        print("❌ No car data files found in data/ directory")
        print("   Please run scrape-cars for some models first")
        return

    print(f"📋 Found {len(available_models)} car models to analyze:")
    for brand, model, normalized_name, _ in available_models:
        print(f"   - {brand} {model}")
    print()

    # Run analysis for each model
    models_with_analysis = []

    for i, (brand, model, normalized_name, data_file) in enumerate(available_models, 1):
        print(f"[{i}/{len(available_models)}] Analyzing {brand} {model}...")

        try:
            # Initialize analyzer for this model
            analyzer = PersonalCarAnalyzer(brand, model)

            # Get best personal deals
            best_deals = analyzer.get_best_personal_deals(
                years_owned=years_owned,
                km_per_year=km_per_year,
                top_n=20,
                **filter_kwargs,
            )

            if len(best_deals) > 0:
                # Generate personal analysis report
                analyzer.generate_personal_report(
                    years_owned=years_owned,
                    km_per_year=km_per_year,
                    top_n=20,
                    **filter_kwargs,
                )

                # Save best deals CSV
                paths = get_brand_model_paths(brand, model)
                best_deals.to_csv(paths["personal_deals"], index=False)

                # Collect analysis data for index
                analysis_data = {
                    "total_listings": (
                        len(analyzer.df) if analyzer.df is not None else 0
                    ),
                    "best_deal_price": (
                        float(best_deals.iloc[0]["price"]) if len(best_deals) > 0 else 0
                    ),
                    "best_deal_cost_per_km": (
                        float(best_deals.iloc[0]["cost_per_km"])
                        if len(best_deals) > 0
                        else 0
                    ),
                    "avg_annual_cost": (
                        float(best_deals["total_annual_cost"].mean())
                        if len(best_deals) > 0
                        else 0
                    ),
                    "has_personal_analysis": True,
                }

                print(
                    f"   ✓ Complete - Best deal: €{analysis_data['best_deal_price']:,.0f} ({analysis_data['best_deal_cost_per_km']:.3f}/km)"
                )
            else:
                analysis_data = {
                    "total_listings": (
                        len(analyzer.df) if analyzer.df is not None else 0
                    ),
                    "has_personal_analysis": False,
                }
                print(f"   ⚠️  No deals found matching criteria")

        except Exception as e:
            print(f"   ❌ Error analyzing {brand} {model}: {str(e)}")
            analysis_data = {"total_listings": 0, "has_personal_analysis": False}

        models_with_analysis.append((brand, model, normalized_name, analysis_data))

    print()
    print("📊 Generating analysis index...")
    generate_analysis_index(models_with_analysis)

    # Summary
    successful_analyses = sum(
        1
        for _, _, _, data in models_with_analysis
        if data.get("has_personal_analysis", False)
    )
    print()
    print("🎉 Comprehensive analysis complete!")
    print(
        f"   ✓ {successful_analyses}/{len(available_models)} models analyzed successfully"
    )
    print(f"   ✓ Index page: reports/index.html")
    print(f"   ✓ All personal analysis reports generated")


def main_comprehensive():
    """Command-line interface for comprehensive analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run personal analysis for all available car models and generate index"
    )
    parser.add_argument(
        "--years-owned",
        type=int,
        default=5,
        help="Number of years you plan to own the car (default: 5)",
    )
    parser.add_argument(
        "--km-per-year",
        type=int,
        default=30000,
        help="Kilometers driven per year (default: 30,000)",
    )
    parser.add_argument("--max-age", type=float, help="Maximum car age in years")
    parser.add_argument("--max-km", type=int, help="Maximum current kilometers")
    parser.add_argument("--max-price", type=int, help="Maximum price")
    parser.add_argument("--min-price", type=int, help="Minimum price")
    parser.add_argument(
        "--fuel-type", type=str, help="Filter by fuel type (e.g., Gasoline, Diesel)"
    )

    args = parser.parse_args()

    # Prepare filter arguments
    filter_kwargs = {}
    if args.max_age:
        filter_kwargs["max_age"] = args.max_age
    if args.max_km:
        filter_kwargs["max_km"] = args.max_km
    if args.max_price:
        filter_kwargs["max_price"] = args.max_price
    if args.min_price:
        filter_kwargs["min_price"] = args.min_price
    if args.fuel_type:
        filter_kwargs["fuel_type"] = args.fuel_type

    run_comprehensive_analysis(
        years_owned=args.years_owned, km_per_year=args.km_per_year, **filter_kwargs
    )


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


def deduplicate(make, model):
    """Command-line interface for the duplicate cleaner."""
    parser = argparse.ArgumentParser(
        description="Clean duplicate car listings from CSV file"
    )
    parser.add_argument("make", help="Car make name (e.g., Peugeot, Volkswagen)")
    parser.add_argument("model", help="Car model name (e.g., 2008, Golf GTI)")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=[
            "kilometers",
            "month_of_registration",
            "year_of_registration",
            "paint",
            "horsepower",
        ],
        help="Columns to use for duplicate detection (default: kilometers age_in_years horsepower)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview duplicates without removing them",
    )

    args = parser.parse_args()

    if not args.make or not args.model:
        args.make = make
        args.model = model

    if not args.make or not args.model:
        print("Error: Please provide both make and model names.")
        return

    # Get standardized file paths
    paths = get_brand_model_paths(args.make, args.model)
    input_file = paths["data_file"]

    # Validate input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist")
        print(f"Expected file for {args.make} {args.model}: {input_file}")
        return

    if args.preview:
        # Preview mode - just show what would be removed
        df = pd.read_csv(input_file)
        print(
            f"Loaded {len(df)} records for {args.make} {args.model} from {input_file}"
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
        print(f"Cleaning duplicates for {args.make} {args.model}...")
        clean_duplicates(
            input_file, input_file, args.columns
        )  # Always overwrite source


if __name__ == "__main__":
    deduplicate()
