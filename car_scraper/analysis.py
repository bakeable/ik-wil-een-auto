"""Statistical analysis and modeling for car value assessment."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")


class CarValueAnalyzer:
    """Analyze car listings and determine best value vehicles."""

    def __init__(self, data_path: str = "data/car_listings.csv"):
        """Initialize with car listing data.

        Args:
            data_path: Path to CSV file with car listing data
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, LabelEncoder] = {}

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess car listing data."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} car listings")

        # Basic data cleaning
        self.df = self.df.dropna(subset=["price", "kilometers", "horsepower"])
        self.df = self.df[self.df["price"] > 0]
        self.df = self.df[self.df["kilometers"] >= 0]
        self.df = self.df[self.df["horsepower"] > 0]

        print(f"After cleaning: {len(self.df)} listings")
        return self.df

    def explore_data(self, save_plots: bool = True) -> Dict[str, any]:
        """Perform exploratory data analysis.

        Args:
            save_plots: Whether to save plots to reports directory

        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            self.load_data()

        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Basic statistics
        stats = {
            "total_listings": len(self.df),
            "price_stats": self.df["price"].describe().to_dict(),
            "mileage_stats": self.df["kilometers"].describe().to_dict(),
            "age_stats": self.df["age_in_years"].describe().to_dict(),
            "horsepower_stats": self.df["horsepower"].describe().to_dict(),
        }

        if save_plots:
            # Price distribution
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.hist(self.df["price"], bins=50, alpha=0.7, edgecolor="black")
            plt.title("Price Distribution")
            plt.xlabel("Price (€)")
            plt.ylabel("Frequency")

            plt.subplot(2, 2, 2)
            plt.scatter(self.df["kilometers"], self.df["price"], alpha=0.5)
            plt.title("Price vs Mileage")
            plt.xlabel("Kilometers")
            plt.ylabel("Price (€)")

            plt.subplot(2, 2, 3)
            plt.scatter(self.df["age_in_years"], self.df["price"], alpha=0.5)
            plt.title("Price vs Age")
            plt.xlabel("Age (years)")
            plt.ylabel("Price (€)")

            plt.subplot(2, 2, 4)
            plt.scatter(self.df["horsepower"], self.df["price"], alpha=0.5)
            plt.title("Price vs Horsepower")
            plt.xlabel("Horsepower")
            plt.ylabel("Price (€)")

            plt.tight_layout()
            plt.savefig(
                reports_dir / "price_analysis.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            # Fuel type analysis
            plt.figure(figsize=(10, 6))
            fuel_counts = self.df["fuel_type"].value_counts()
            plt.subplot(1, 2, 1)
            fuel_counts.plot(kind="bar")
            plt.title("Fuel Type Distribution")
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            fuel_prices = (
                self.df.groupby("fuel_type")["price"]
                .mean()
                .sort_values(ascending=False)
            )
            fuel_prices.plot(kind="bar", color="orange")
            plt.title("Average Price by Fuel Type")
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(reports_dir / "fuel_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()

        return stats

    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning model.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.df is None:
            self.load_data()

        # Select features for modeling
        feature_columns = [
            "kilometers",
            "age_in_years",
            "horsepower",
            "fuel_type",
            "gearbox",
            "co2_emissions",
        ]

        # Create working copy
        df_model = self.df.copy()

        # Handle missing values
        df_model["gearbox"] = df_model["gearbox"].fillna("Unknown")
        if "co2_emissions" in df_model.columns:
            df_model["co2_emissions"] = pd.to_numeric(
                df_model["co2_emissions"], errors="coerce"
            ).fillna(0)
        else:
            df_model["co2_emissions"] = 0

        # Encode categorical variables
        categorical_cols = ["fuel_type", "gearbox"]
        for col in categorical_cols:
            if col in df_model.columns:
                le = LabelEncoder()
                df_model[col + "_encoded"] = le.fit_transform(df_model[col].astype(str))
                self.encoders[col] = le

        # Select final features
        model_features = [
            "kilometers",
            "age_in_years",
            "horsepower",
            "fuel_type_encoded",
            "gearbox_encoded",
            "co2_emissions",
        ]

        X = df_model[model_features].fillna(0)
        y = df_model["price"]

        return X, y

    def train_model(self) -> Dict[str, float]:
        """Train price prediction model.

        Returns:
            Dictionary with model performance metrics
        """
        X, y = self.prepare_features()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

        print("Model Performance:")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"RMSE: €{metrics['rmse']:.0f}")
        print(f"MAE: €{metrics['mae']:.0f}")

        return metrics

    def calculate_value_score(self) -> pd.DataFrame:
        """Calculate value score for each car.

        Returns:
            DataFrame with value scores and recommendations
        """
        if self.model is None:
            self.train_model()

        if self.df is None or self.model is None or self.scaler is None:
            raise ValueError("Model not trained or data not available")

        X, _ = self.prepare_features()
        X_scaled = self.scaler.transform(X)

        # Predict prices
        predicted_prices = self.model.predict(X_scaled)

        # Calculate value score (actual vs predicted price)
        self.df["predicted_price"] = predicted_prices
        self.df["price_difference"] = self.df["predicted_price"] - self.df["price"]
        self.df["value_score"] = (
            self.df["price_difference"] / self.df["predicted_price"] * 100
        )  # Categorize deals

        def categorize_deal(score):
            if score >= 15:
                return "Excellent Deal"
            elif score >= 5:
                return "Good Deal"
            elif score >= -5:
                return "Fair Deal"
            else:
                return "Overpriced"

        self.df["deal_category"] = self.df["value_score"].apply(categorize_deal)

        return self.df[
            ["detail_url", "price", "predicted_price", "value_score", "deal_category"]
        ]

    def get_best_deals(
        self,
        top_n: int = 20,
        max_km: int = None,
        max_age: float = None,
        min_price: int = None,
        max_price: int = None,
        fuel_type: str = None,
    ) -> pd.DataFrame:
        """Get the top N best value cars with optional filtering.

        Args:
            top_n: Number of top deals to return
            max_km: Maximum kilometers for filtering results
            max_age: Maximum age in years for filtering results
            min_price: Minimum price for filtering results
            max_price: Maximum price for filtering results
            fuel_type: Specific fuel type for filtering results

        Returns:
            DataFrame with best deals matching criteria
        """
        if self.df is None:
            raise ValueError("No data available")

        if "value_score" not in self.df.columns:
            self.calculate_value_score()

        # Start with all data (for proper model training)
        filtered_deals = self.df.copy()

        # Apply filters to results only (not to training data)
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

        # Get top deals from filtered results
        best_deals = filtered_deals.nlargest(top_n, "value_score")

        print(
            f"Found {len(best_deals)} best deals from {len(filtered_deals)} cars matching criteria"
        )

        columns_to_show = [
            "detail_url",
            "price",
            "predicted_price",
            "value_score",
            "kilometers",
            "age_in_years",
            "horsepower",
            "fuel_type",
            "deal_category",
        ]

        return best_deals[columns_to_show]

    def filter_cars(
        self,
        max_age: float = None,
        min_price: int = None,
        max_price: int = None,
        fuel_type: str = None,
        max_km: int = None,
    ) -> pd.DataFrame:
        """Filter cars based on various criteria.

        Args:
            max_age: Maximum age in years
            min_price: Minimum price in euros
            max_price: Maximum price in euros
            fuel_type: Specific fuel type to filter by
            max_km: Maximum kilometers

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            self.load_data()

        filtered_df = self.df.copy()

        if max_age is not None:
            filtered_df = filtered_df[filtered_df["age_in_years"] < max_age]

        if min_price is not None:
            filtered_df = filtered_df[filtered_df["price"] >= min_price]

        if max_price is not None:
            filtered_df = filtered_df[filtered_df["price"] <= max_price]

        if fuel_type is not None:
            filtered_df = filtered_df[filtered_df["fuel_type"] == fuel_type]

        if max_km is not None:
            filtered_df = filtered_df[filtered_df["kilometers"] <= max_km]

        print(f"Found {len(filtered_df)} cars matching your criteria")
        return filtered_df

    def _format_table_with_clickable_urls(self, df: pd.DataFrame) -> str:
        """Format DataFrame as HTML table with clickable URLs."""
        # Create a copy to avoid modifying original
        display_df = df.copy()

        # Format detail_url as clickable link
        if "detail_url" in display_df.columns:
            display_df["detail_url"] = display_df["detail_url"].apply(
                lambda url: f'<a href="{url}" target="_blank">View Car</a>'
            )

        # Format numeric columns nicely
        if "price" in display_df.columns:
            display_df["price"] = display_df["price"].apply(lambda x: f"€{x:,.0f}")
        if "predicted_price" in display_df.columns:
            display_df["predicted_price"] = display_df["predicted_price"].apply(
                lambda x: f"€{x:,.0f}"
            )
        if "value_score" in display_df.columns:
            display_df["value_score"] = display_df["value_score"].apply(
                lambda x: f"{x:.1f}%"
            )

        return display_df.to_html(classes="table", escape=False, index=False)

    def generate_report(
        self, output_path: str = "reports/analysis_report.html"
    ) -> None:
        """Generate comprehensive analysis report.

        Args:
            output_path: Path to save HTML report
        """
        # Ensure analysis is complete
        stats = self.explore_data()
        self.train_model()
        best_deals = self.get_best_deals()

        # Make URLs clickable by converting them to HTML links
        best_deals_display = (
            best_deals[
                (best_deals["kilometers"] < 125000) & (best_deals["price"] < 16500)
            ]
            .head(20)
            .copy()
        )
        best_deals_display["detail_url"] = best_deals_display["detail_url"].apply(
            lambda url: f'<a href="{url}" target="_blank">View Listing</a>'
        )
        best_deals_display["price"] = best_deals_display["price"].apply(
            lambda x: f"€{x:,.0f}"
        )
        best_deals_display["predicted_price"] = best_deals_display[
            "predicted_price"
        ].apply(lambda x: f"€{x:,.0f}")
        best_deals_display["value_score"] = best_deals_display["value_score"].apply(
            lambda x: f"{x:.1f}%"
        )
        best_deals_display = best_deals_display.rename(
            columns={
                "detail_url": "Listing",
                "price": "Actual Price",
                "predicted_price": "Predicted Price",
                "value_score": "Value Score",
                "kilometers": "Kilometers",
                "age_in_years": "Age (years)",
                "horsepower": "Horsepower",
                "fuel_type": "Fuel Type",
                "deal_category": "Deal Category",
            }
        )

        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Car Value Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .excellent {{ background-color: #d4edda; }}
                .good {{ background-color: #d1ecf1; }}
                .fair {{ background-color: #fff3cd; }}
                .overpriced {{ background-color: #f8d7da; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Car Value Analysis Report</h1>
            
            <h2>Dataset Overview</h2>
            <div class="metric">
                <p><strong>Total Listings Analyzed:</strong> {stats['total_listings']}</p>
                <p><strong>Average Price:</strong> €{stats['price_stats']['mean']:.0f}</p>
                <p><strong>Average Mileage:</strong> {stats['mileage_stats']['mean']:.0f} km</p>
                <p><strong>Average Age:</strong> {stats['age_stats']['mean']:.1f} years</p>
            </div>
            
            <h2>Top 10 Best Value Cars</h2>
            {best_deals_display.to_html(classes='table', escape=False)}
            
            <h2>Deal Categories Distribution</h2>
            <p>Excellent Deals: {len(self.df[self.df['deal_category'] == 'Excellent Deal'])}</p>
            <p>Good Deals: {len(self.df[self.df['deal_category'] == 'Good Deal'])}</p>
            <p>Fair Deals: {len(self.df[self.df['deal_category'] == 'Fair Deal'])}</p>
            <p>Overpriced: {len(self.df[self.df['deal_category'] == 'Overpriced'])}</p>
        </body>
        </html>
        """

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Analysis report saved to {output_path}")


def main():
    """Main entry point for analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze car value from scraped data")
    parser.add_argument(
        "--data", default="data/listings.csv", help="Path to CSV data file"
    )
    parser.add_argument(
        "--report", default="reports/analysis_report.html", help="Output report path"
    )
    parser.add_argument(
        "--top-deals", type=int, default=20, help="Number of top deals to show"
    )
    parser.add_argument(
        "--max-age", type=float, help="Filter cars with maximum age in years"
    )
    parser.add_argument("--max-price", type=int, help="Filter cars with maximum price")
    parser.add_argument(
        "--fuel-type", type=str, help="Filter by fuel type (e.g., Gasoline, Diesel)"
    )
    parser.add_argument(
        "--max-km", type=int, help="Filter cars with maximum kilometers"
    )
    parser.add_argument(
        "--deals-only",
        action="store_true",
        help="Apply filters to results only, not to model training data",
    )

    args = parser.parse_args()

    analyzer = CarValueAnalyzer(args.data)

    try:
        # Run full analysis
        print("Starting car value analysis...")

        # Apply filters if specified
        if (
            args.max_age or args.max_price or args.fuel_type or args.max_km
        ) and not args.deals_only:
            print("Applying filters to training data...")
            filtered_data = analyzer.filter_cars(
                max_age=args.max_age,
                max_price=args.max_price,
                fuel_type=args.fuel_type,
                max_km=args.max_km,
            )
            # Update analyzer's dataframe for subsequent analysis
            analyzer.df = filtered_data

        stats = analyzer.explore_data()
        print("✓ Data exploration complete")

        metrics = analyzer.train_model()
        print("✓ Price prediction model trained")

        # Get best deals with optional filtering
        if args.deals_only and (
            args.max_age or args.max_price or args.fuel_type or args.max_km
        ):
            print("Applying filters to results only...")
            best_deals = analyzer.get_best_deals(
                top_n=args.top_deals,
                max_age=args.max_age,
                max_price=args.max_price,
                fuel_type=args.fuel_type,
                max_km=args.max_km,
            )
        else:
            best_deals = analyzer.get_best_deals(args.top_deals)

        print(f"✓ Found {len(best_deals)} best deals")

        analyzer.generate_report(args.report)
        print("✓ Analysis report generated")

        # Save best deals to CSV
        best_deals_path = "reports/best_deals.csv"
        best_deals.to_csv(best_deals_path, index=False)
        print(f"✓ Best deals saved to {best_deals_path}")

    except FileNotFoundError:
        print("Error: No data file found. Please run the scraper first.")
        print("Usage: poetry run scrape-cars")


if __name__ == "__main__":
    main()
