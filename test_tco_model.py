#!/usr/bin/env python3
"""Test script to verify the realistic TCO model is working correctly."""

import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from car_scraper.personal_analysis import PersonalCarAnalyzer


def test_realistic_tco_model():
    """Test the realistic TCO model and compare high vs low mileage cars."""

    print("=" * 60)
    print("TESTING REALISTIC TCO MODEL")
    print("=" * 60)

    # Test parameters
    years_owned = 4
    km_per_year = 40000

    analyzer = PersonalCarAnalyzer("data/listings.csv")

    print(f"\nTesting with {years_owned} years ownership, {km_per_year:,} km/year")
    print("-" * 50)

    try:
        # Get best deals
        best_deals = analyzer.get_best_personal_deals(
            years_owned=years_owned,
            km_per_year=km_per_year,
            top_n=10,
            max_age=8,  # Allow a range of ages
            fuel_type="Gasoline",
        )

        print(f"✓ Successfully analyzed {len(best_deals)} cars")

        # Check that all required columns exist
        required_columns = [
            "price",
            "future_value",
            "total_depreciation",
            "cost_per_km",
            "annual_running_cost",
            "estimated_annual_maintenance",
            "kilometers",
            "age_in_years",
        ]

        missing_cols = [
            col for col in required_columns if col not in best_deals.columns
        ]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print("✓ All required columns present")

        # Verify depreciation is never negative
        min_depreciation = best_deals["total_depreciation"].min()
        print(f"✓ Minimum depreciation: €{min_depreciation:,.0f} (should be >= 0)")

        if min_depreciation < 0:
            print("❌ Found negative depreciation!")
            return False

        # Show comparison of high vs low mileage cars
        print("\n" + "=" * 60)
        print("HIGH VS LOW MILEAGE COMPARISON")
        print("=" * 60)

        # Find low mileage cars (< 100k km)
        low_km_cars = best_deals[best_deals["kilometers"] < 100000].head(3)
        print(f"\nTOP 3 LOW MILEAGE CARS (<100k km):")
        print("-" * 40)
        for i, car in low_km_cars.iterrows():
            print(f"🚗 {car['age_in_years']:.1f}y, {car['kilometers']:,}km")
            print(f"   Price: €{car['price']:,} → Resale: €{car['future_value']:,.0f}")
            print(f"   Maintenance/year: €{car['estimated_annual_maintenance']:,.0f}")
            print(f"   TOTAL COST/KM: €{car['cost_per_km']:.3f}")
            print()

        # Find high mileage cars (> 150k km) if any made it to top deals
        high_km_cars = best_deals[best_deals["kilometers"] > 150000].head(3)
        if len(high_km_cars) > 0:
            print(f"\nTOP 3 HIGH MILEAGE CARS (>150k km) that made it to best deals:")
            print("-" * 40)
            for i, car in high_km_cars.iterrows():
                print(f"🚗 {car['age_in_years']:.1f}y, {car['kilometers']:,}km")
                print(
                    f"   Price: €{car['price']:,} → Resale: €{car['future_value']:,.0f}"
                )
                print(
                    f"   Maintenance/year: €{car['estimated_annual_maintenance']:,.0f}"
                )
                print(f"   TOTAL COST/KM: €{car['cost_per_km']:.3f}")
                print()
        else:
            print("\n✓ NO HIGH MILEAGE CARS in top deals - model working correctly!")

        # Overall statistics
        print("\n" + "=" * 60)
        print("OVERALL TCO STATISTICS")
        print("=" * 60)

        avg_km = best_deals["kilometers"].mean()
        avg_age = best_deals["age_in_years"].mean()
        avg_cost_per_km = best_deals["cost_per_km"].mean()
        avg_maintenance = best_deals["estimated_annual_maintenance"].mean()

        print(f"Average kilometers: {avg_km:,.0f}")
        print(f"Average age: {avg_age:.1f} years")
        print(f"Average cost per km: €{avg_cost_per_km:.3f}")
        print(f"Average annual maintenance: €{avg_maintenance:,.0f}")

        # Check if model properly penalizes high mileage
        mileage_ranges = [
            (0, 80000, "Low"),
            (80000, 150000, "Medium"),
            (150000, 300000, "High"),
        ]

        print(f"\nMAINTENANCE COSTS BY MILEAGE RANGE:")
        print("-" * 40)
        for min_km, max_km, label in mileage_ranges:
            range_cars = best_deals[
                (best_deals["kilometers"] >= min_km)
                & (best_deals["kilometers"] < max_km)
            ]
            if len(range_cars) > 0:
                avg_maint = range_cars["estimated_annual_maintenance"].mean()
                print(
                    f"{label} mileage ({min_km/1000:.0f}k-{max_km/1000:.0f}k): €{avg_maint:,.0f}/year avg"
                )

        print("\n✅ TCO MODEL TEST COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_realistic_tco_model()
    sys.exit(0 if success else 1)
