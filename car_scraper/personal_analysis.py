"""Personal cost analysis for car ownership decisions."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .analysis import CarValueAnalyzer
from .utils import deduplicate, get_brand_model_paths


class PersonalCarAnalyzer(CarValueAnalyzer):
    """Analyze car listings for personal cost-effectiveness based on individual usage patterns."""

    def calculate_future_value(
        self, years_owned: int, km_per_year: int
    ) -> pd.DataFrame:
        """Calculate predicted future value after specified ownership period.

        Args:
            years_owned: Number of years you plan to own the car
            km_per_year: Kilometers driven per year

        Returns:
            DataFrame with future values calculated
        """
        if self.df is None:
            self.load_data()

        if self.ml_model is None or self.scaler is None:
            self.train_model()

        # Calculate future car specs after ownership period
        future_df = self.df.copy()
        future_df["future_age"] = future_df["age_in_years"] + years_owned
        future_df["future_km"] = future_df["kilometers"] + (years_owned * km_per_year)

        # Prepare features for future value prediction
        future_features = future_df.copy()

        # Handle missing values same as training
        future_features["gearbox"] = future_features["gearbox"].fillna("Unknown")
        if "co2_emissions" in future_features.columns:
            future_features["co2_emissions"] = pd.to_numeric(
                future_features["co2_emissions"], errors="coerce"
            ).fillna(0)
        else:
            future_features["co2_emissions"] = 0

        # Encode categorical variables using existing encoders
        for col, encoder in self.encoders.items():
            if col in future_features.columns:
                # Handle unknown categories by using most frequent class
                future_features[col + "_encoded"] = (
                    future_features[col]
                    .astype(str)
                    .map(
                        dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    )
                    .fillna(encoder.transform([encoder.classes_[0]])[0])
                )

        # Create feature matrix for future prediction
        model_features = [
            "future_km",  # Use future kilometers
            "future_age",  # Use future age
            "horsepower",
            "fuel_type_encoded",
            "gearbox_encoded",
            "co2_emissions",
        ]

        # Rename columns to match training feature names
        feature_mapping = {"future_km": "kilometers", "future_age": "age_in_years"}

        X_future = (
            future_features[model_features].rename(columns=feature_mapping).fillna(0)
        )
        X_future_scaled = self.scaler.transform(X_future)

        # Predict future values
        future_values = self.ml_model.predict(X_future_scaled)
        self.df["future_value"] = future_values

        return self.df

    def _estimate_fuel_consumption_l_per_100km(self, row: pd.Series) -> float:
        """Return estimated fuel consumption (L/100km) for a given car."""
        # If you have a column like 'consumption_l_per_100km', prefer that
        for col in ["consumption_l_per_100km", "consumption", "avg_consumption"]:
            if col in row and pd.notna(row[col]):
                try:
                    return float(row[col])
                except (TypeError, ValueError):
                    pass

        # Fallback defaults per fuel type
        fuel = str(row.get("fuel_type", "")).lower()
        if "electric" in fuel:
            # Convert kWh/100km to L/100km-equivalent or just return a fake low number
            return 0.0
        if "diesel" in fuel:
            return 6.5
        if "hybrid" in fuel or "plug-in" in fuel:
            return 5.5
        # default petrol
        return 7.5

    def _estimate_annual_fuel_cost(self, row: pd.Series, km_per_year: int) -> float:
        """Calculate realistic annual fuel cost based on car-specific consumption."""
        l_per_100km = self._estimate_fuel_consumption_l_per_100km(row)
        fuel_price = 2.10  # € per liter, make configurable
        return (km_per_year / 100) * l_per_100km * fuel_price

    def _estimate_annual_insurance(self, row: pd.Series) -> float:
        """Realistic insurance model with age discounts."""
        price = float(row["price"])
        age = float(row["age_in_years"])

        # Base WA premium
        base = 350.0

        # Value-based component (~1.2% of value instead of 2%)
        value_component = 0.012 * price

        # Older cars often get cheaper insurance (less coverage, WA ipv AllRisk)
        age_discount_factor = max(
            0.4, 1.0 - (age / 12.0)
        )  # at 12+ yrs, 60% of original

        premium = base + value_component * age_discount_factor

        # Clamp to sensible bounds
        return float(np.clip(premium, 350, 1200))

    def _estimate_annual_maintenance(self, row: pd.Series, km_per_year: int) -> float:
        """Maintenance + wear + expected repairs based on age & mileage."""
        age = float(row["age_in_years"])
        km_current = float(row["kilometers"])

        # Base maintenance for a relatively young, low-km car
        base_maintenance = 600.0  # annual service, minor repairs

        # Variable maintenance roughly per km (tyres, brakes, etc.)
        variable_per_km = 0.05  # € per km
        variable_component = km_per_year * variable_per_km

        # Age factor: after ~5 years, maintenance climbs
        age_penalty = max(0, age - 5) * 120.0  # € per extra year over 5

        # Mileage levels: penalty per year based on *current* mileage
        mileage_penalty = 0.0
        if km_current > 250_000:
            mileage_penalty = 1500.0
        elif km_current > 200_000:
            mileage_penalty = 1000.0
        elif km_current > 150_000:
            mileage_penalty = 600.0
        elif km_current > 100_000:
            mileage_penalty = 350.0
        elif km_current > 80_000:
            mileage_penalty = 150.0

        # Expected major repair risk (gearbox, turbo, clutch, etc.)
        # This is a more realistic probability model.
        risk_score = 0.0
        if km_current > 200_000:
            risk_score += 0.20  # 20% base risk for very high mileage
        elif km_current > 150_000:
            risk_score += 0.12  # 12% base risk
        elif km_current > 100_000:
            risk_score += 0.06  # 6% base risk
        elif km_current > 80_000:
            risk_score += 0.03  # 3% base risk

        if age > 8:
            risk_score += (age - 8) * 0.04  # 4% per year over 8
        elif age > 5:
            risk_score += (age - 5) * 0.02  # 2% per year 5-8

        risk_score = float(
            np.clip(risk_score, 0.0, 0.6)
        )  # cap at 60% per year for extreme cases
        average_major_repair_cost = 1_800.0
        expected_major_repairs = risk_score * average_major_repair_cost

        return (
            base_maintenance
            + variable_component
            + age_penalty
            + mileage_penalty
            + expected_major_repairs
        )

    def calculate_personal_cost_analysis(
        self, years_owned: int, km_per_year: int
    ) -> pd.DataFrame:
        """Calculate realistic personalized cost analysis with proper TCO model.

        Args:
            years_owned: Number of years you plan to own the car
            km_per_year: Kilometers driven per year

        Returns:
            DataFrame with comprehensive cost analysis
        """
        if self.df is None:
            self.load_data()

        # Calculate standard market value analysis first
        self.calculate_value_score()

        # Calculate predicted resale value after ownership
        self.calculate_future_value(years_owned, km_per_year)

        total_km = years_owned * km_per_year

        # Depreciation = Purchase Price - Future Resale Value
        # Ensure depreciation is never negative (car can't appreciate)
        self.df["total_depreciation"] = (
            self.df["price"] - self.df["future_value"]
        ).clip(lower=0)

        # Per-year depreciation for reporting only
        self.df["annual_depreciation"] = self.df["total_depreciation"] / years_owned

        # Fuel, insurance, maintenance – computed per year per car using realistic models
        self.df["estimated_annual_fuel"] = self.df.apply(
            lambda row: self._estimate_annual_fuel_cost(row, km_per_year), axis=1
        )

        self.df["estimated_annual_insurance"] = self.df.apply(
            self._estimate_annual_insurance, axis=1
        )

        self.df["estimated_annual_maintenance"] = self.df.apply(
            lambda row: self._estimate_annual_maintenance(row, km_per_year), axis=1
        )

        # Running costs per year (non-depreciation)
        self.df["annual_running_cost"] = (
            self.df["estimated_annual_fuel"]
            + self.df["estimated_annual_insurance"]
            + self.df["estimated_annual_maintenance"]
        )

        # Total TCO over the whole period
        self.df["total_running_cost"] = self.df["annual_running_cost"] * years_owned
        self.df["total_cost_of_ownership"] = (
            self.df["total_depreciation"] + self.df["total_running_cost"]
        )

        # Cost per km over the full horizon (the key metric!)
        self.df["cost_per_km"] = self.df["total_cost_of_ownership"] / total_km

        # For convenience, still expose "total_annual_cost" as before (now derived)
        self.df["total_annual_cost"] = self.df["total_cost_of_ownership"] / years_owned

        return self.df

    def get_best_personal_deals(
        self,
        years_owned: int,
        km_per_year: int,
        top_n: int = 20,
        max_km: int = None,
        max_age: float = None,
        min_price: int = None,
        max_price: int = None,
        fuel_type: str = None,
    ) -> pd.DataFrame:
        """Get the best personal deals based on total cost of ownership.

        Args:
            years_owned: Number of years you plan to own the car
            km_per_year: Kilometers driven per year
            top_n: Number of top deals to return
            max_km: Maximum current kilometers for filtering
            max_age: Maximum current age for filtering
            min_price: Minimum price for filtering
            max_price: Maximum price for filtering
            fuel_type: Specific fuel type for filtering

        Returns:
            DataFrame with best personal deals sorted by total annual cost
        """
        if self.df is None:
            self.load_data()

        # Calculate personal cost analysis
        self.calculate_personal_cost_analysis(years_owned, km_per_year)

        # Apply filters
        filtered_deals = self.df.copy()

        if max_km is not None:
            filtered_deals = filtered_deals[filtered_deals["kilometers"] <= max_km]
        if max_age is not None:
            filtered_deals = filtered_deals[filtered_deals["age_in_years"] < max_age]
        if min_price is not None:
            filtered_deals = filtered_deals[filtered_deals["price"] >= min_price]
        if max_price is not None:
            filtered_deals = filtered_deals[filtered_deals["price"] <= max_price]
        if fuel_type is not None:
            filtered_deals = filtered_deals[filtered_deals["fuel_type"] == fuel_type]

        # Sort by cost per km (lowest = best deal) - the true TCO metric
        best_personal_deals = filtered_deals.nsmallest(top_n, "cost_per_km")

        print(
            f"Found {len(best_personal_deals)} best personal deals for {years_owned} years ownership, "
            f"{km_per_year:,} km/year from {len(filtered_deals)} cars matching criteria"
        )

        columns_to_show = [
            "detail_url",
            "price",
            "predicted_price",
            "value_score",
            "future_value",
            "total_depreciation",
            "total_cost_of_ownership",
            "estimated_annual_fuel",
            "estimated_annual_insurance",
            "estimated_annual_maintenance",
            "annual_depreciation",
            "annual_running_cost",
            "total_annual_cost",
            "cost_per_km",
            "kilometers",
            "age_in_years",
            "horsepower",
            "fuel_type",
        ]

        return best_personal_deals[columns_to_show]

    def generate_personal_report(
        self,
        years_owned: int,
        km_per_year: int,
        output_path: str = None,
        top_n: int = 20,
        **filter_kwargs,
    ) -> None:
        """Generate comprehensive personal cost analysis report.

        Args:
            years_owned: Number of years you plan to own the car
            km_per_year: Kilometers driven per year
            output_path: Path to save HTML report (auto-generated if not provided)
            top_n: Number of cars to show in report
            **filter_kwargs: Additional filters (max_km, max_age, etc.)
        """
        # Determine output path
        if output_path is None:
            if hasattr(self, "paths") and self.paths:
                output_path = self.paths["personal_report"]
            else:
                output_path = "reports/personal_analysis_report.html"
        # Ensure analysis is complete
        stats = self.explore_data()
        self.train_model()

        # Get best personal deals
        best_deals = self.get_best_personal_deals(
            years_owned=years_owned,
            km_per_year=km_per_year,
            top_n=top_n,
            **filter_kwargs,
        )

        # Calculate statistics for report
        avg_annual_cost = best_deals["total_annual_cost"].mean()
        avg_cost_per_km = best_deals["cost_per_km"].mean()
        avg_depreciation = best_deals["annual_depreciation"].mean()

        # Make URLs clickable
        best_deals_display = best_deals.copy()
        best_deals_display["detail_url"] = best_deals_display["detail_url"].apply(
            lambda url: f'<a href="{url}" target="_blank">View Listing</a>'
        )

        # Format currency columns
        currency_columns = [
            "price",
            "predicted_price",
            "future_value",
            "total_depreciation",
            "total_cost_of_ownership",
            "annual_depreciation",
            "estimated_annual_fuel",
            "estimated_annual_insurance",
            "estimated_annual_maintenance",
            "total_annual_cost",
        ]
        for col in currency_columns:
            if col in best_deals_display.columns:
                best_deals_display[col] = best_deals_display[col].apply(
                    lambda x: f"€{x:,.0f}"
                )

        if "cost_per_km" in best_deals_display.columns:
            best_deals_display["cost_per_km"] = best_deals_display["cost_per_km"].apply(
                lambda x: f"€{x:.3f}"
            )

        if "value_score" in best_deals_display.columns:
            best_deals_display["value_score"] = best_deals_display["value_score"].apply(
                lambda x: f"{x:.1f}%"
            )

        best_deals_display = best_deals_display.rename(
            columns={
                "detail_url": "Listing",
                "price": "Price",
                "predicted_price": "Predicted Market Value",
                "value_score": "Market Value Score",
                "future_value": "Predicted Resale Value",
                "total_depreciation": "Total Depreciation",
                "total_cost_of_ownership": "Total Cost of Ownership",
                "annual_depreciation": "Annual Depreciation",
                "annual_running_cost": "Annual Running Cost",
                "estimated_annual_fuel": "Estimated Annual Fuel Cost",
                "estimated_annual_insurance": "Estimated Annual Insurance",
                "estimated_annual_maintenance": "Estimated Annual Maintenance",
                "total_annual_cost": "Total Annual Cost",
                "cost_per_km": "Cost per Kilometer",
                "kilometers": "Current Kilometers",
                "age_in_years": "Age (Years)",
                "horsepower": "Horsepower",
                "fuel_type": "Fuel Type",
            }
        )

        # Generate HTML report
        title_suffix = (
            f" - {self.brand} {self.car_model}" if self.brand and self.car_model else ""
        )
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Personal Car Cost Analysis - {years_owned} Years, {km_per_year:,} km/year{title_suffix}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .highlight {{ background-color: #d4edda; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .back-button {{ 
                    display: inline-block;
                    padding: 10px 20px;
                    background: #6c757d;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    margin: 20px 0;
                    font-weight: bold;
                }}
                .back-button:hover {{ 
                    background: #5a6268;
                    color: white;
                }}
                .header-nav {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding: 10px 0;
                    border-bottom: 1px solid #dee2e6;
                }}
            </style>
        </head>
        <body>
            <div class="header-nav">
                <h1>Personal Car Cost Analysis{title_suffix}</h1>
                <a href="index.html" class="back-button">🏠 Back to Dashboard</a>
            </div>
            
            <h2>Your Usage Profile</h2>
            <div class="metric">
                <p><strong>Ownership Period:</strong> {years_owned} years</p>
                <p><strong>Annual Driving:</strong> {km_per_year:,} km/year</p>
                <p><strong>Total Expected Driving:</strong> {years_owned * km_per_year:,} km</p>
            </div>
            
                        <h2>Realistic TCO Analysis Summary (Top {len(best_deals)} Cars)</h2>
            <div class="metric highlight">
                <p><strong>Average Total Cost of Ownership:</strong> €{best_deals['total_cost_of_ownership'].mean():.0f} over {years_owned} years</p>
                <p><strong>Average Cost per Kilometer:</strong> €{avg_cost_per_km:.3f}</p>
                <p><strong>Average Annual Depreciation:</strong> €{avg_depreciation:.0f}</p>
                <p><strong>Average Annual Running Costs:</strong> €{best_deals['annual_running_cost'].mean():.0f}</p>
            </div>
            
            <h2>Dataset Overview</h2>
            <div class="metric">
                <p><strong>Total Listings Analyzed:</strong> {stats['total_listings']}</p>
                <p><strong>Average Market Price:</strong> €{stats['price_stats']['mean']:.0f}</p>
                <p><strong>Average Mileage:</strong> {stats['mileage_stats']['mean']:.0f} km</p>
                <p><strong>Average Age:</strong> {stats['age_stats']['mean']:.1f} years</p>
            </div>
            
            <h2>Most Cost-Effective Cars for Your Usage</h2>
            {best_deals_display.to_html(classes='table', escape=False, index=False)}
            
            <h2>Realistic TCO Model Explanation</h2>
            <div class="metric">
                <p><strong>Total Cost of Ownership</strong> includes:</p>
                <ul>
                    <li><strong>Depreciation:</strong> Purchase Price - Predicted Resale Value (capped at purchase price)</li>
                    <li><strong>Fuel Costs:</strong> Car-specific consumption rates × {km_per_year:,} km/year</li>
                    <li><strong>Insurance:</strong> Realistic model with age discounts (€350-1200 range)</li>
                    <li><strong>Maintenance:</strong> Base service + age penalties + high-mileage penalties + expected repair risks</li>
                </ul>
                <p><strong>Key Improvements:</strong></p>
                <ul>
                    <li>High-mileage cars (>100k, >150k, >200k km) get significant maintenance penalties</li>
                    <li>Cars >5 years old get increasing age-related repair risk</li>
                    <li>Major repair probability increases with age and mileage</li>
                    <li>Insurance decreases with age (older cars often have basic coverage)</li>
                    <li><strong>Cost per km</strong> is calculated over total ownership period, not just annually</li>
                </ul>
            </div>
        </body>
        </html>
        """

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Personal analysis report saved to {output_path}")


def main():
    """Main entry point for personal analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Personal car cost analysis based on your driving habits"
    )
    parser.add_argument("--make", help="Car brand name (e.g., Peugeot, Volkswagen)")
    parser.add_argument("--model", help="Car model name (e.g., 2008, Golf GTI)")
    parser.add_argument(
        "--years-owned",
        type=int,
        required=True,
        help="Number of years you plan to own the car",
    )
    parser.add_argument(
        "--km-per-year", type=int, required=True, help="Kilometers driven per year"
    )
    parser.add_argument(
        "--top-deals", type=int, default=20, help="Number of top deals to show"
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

    # Deduplicate data
    deduplicate(args.make, args.model)

    # Get file paths for this brand/model
    paths = get_brand_model_paths(args.make, args.model)

    analyzer = PersonalCarAnalyzer(args.make, args.model)

    try:
        # Check if data file exists
        if not Path(paths["data_file"]).exists():
            print(f"Error: No data file found for {args.make} {args.model}")
            print(f"Expected file: {paths['data_file']}")
            print("Please run the scraper first for this brand/model.")
            return

        print(
            f"Starting personal cost analysis for {args.make} {args.model}: {args.years_owned} years, {args.km_per_year:,} km/year..."
        )

        # Get best personal deals
        best_deals = analyzer.get_best_personal_deals(
            years_owned=args.years_owned,
            km_per_year=args.km_per_year,
            top_n=args.top_deals,
            **filter_kwargs,
        )

        # Generate report
        analyzer.generate_personal_report(
            years_owned=args.years_owned,
            km_per_year=args.km_per_year,
            top_n=args.top_deals,
            **filter_kwargs,
        )

        # Save best deals to CSV
        best_deals.to_csv(paths["personal_deals"], index=False)
        print(f"✓ Best personal deals saved to {paths['personal_deals']}")
        print(f"✓ Personal analysis report generated: {paths['personal_report']}")

    except FileNotFoundError:
        print(f"Error: No data file found for {args.brand} {args.model}")
        print(f"Expected file: {paths['data_file']}")
        print("Please run the scraper first for this brand/model.")


if __name__ == "__main__":
    main()
