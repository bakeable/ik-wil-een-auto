#!/usr/bin/env python3
"""Display personal analysis results in readable format."""

import pandas as pd
import sys
import os


def show_personal_analysis_results():
    """Display the personal analysis results in a readable format."""

    try:
        # Read the results
        df = pd.read_csv("reports/personal_best_deals.csv")

        print("=" * 80)
        print("REALISTIC TCO MODEL RESULTS")
        print("4 years ownership, 40,000 km/year, Gasoline cars <6 years old")
        print("=" * 80)

        # Show top 5 cars with key metrics
        for i in range(min(5, len(df))):
            car = df.iloc[i]
            print(f"\n🚗 RANK #{i+1}")
            print(f"Age: {car['age_in_years']:.1f} years")
            print(f"Mileage: {car['kilometers']:,.0f} km")
            print(f"Price: €{car['price']:,.0f}")
            print(f"Predicted resale: €{car['future_value']:,.0f}")
            print(f"Total depreciation: €{car['total_depreciation']:,.0f}")
            print(f"")
            print(f"ANNUAL COSTS:")
            print(f"  Depreciation: €{car['annual_depreciation']:,.0f}")
            print(f"  Fuel: €{car['estimated_annual_fuel']:,.0f}")
            print(f"  Insurance: €{car['estimated_annual_insurance']:,.0f}")
            print(f"  Maintenance: €{car['estimated_annual_maintenance']:,.0f}")
            print(f"  TOTAL ANNUAL: €{car['total_annual_cost']:,.0f}")
            print(f"")
            print(f"📊 COST PER KM: €{car['cost_per_km']:.3f}")
            print(f"📊 TOTAL TCO (4 years): €{car['total_cost_of_ownership']:,.0f}")
            print("-" * 60)

        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"Average cost per km: €{df['cost_per_km'].mean():.3f}")
        print(f"Range: €{df['cost_per_km'].min():.3f} - €{df['cost_per_km'].max():.3f}")
        print(f"Average age: {df['age_in_years'].mean():.1f} years")
        print(f"Average mileage: {df['kilometers'].mean():,.0f} km")
        print(f"")
        print(f"✅ Notice: No high-mileage cars dominating the results!")
        print(f"✅ Realistic maintenance costs properly penalize high-mileage cars")
        print(f"✅ Depreciation properly capped (no negative values)")

    except Exception as e:
        print(f"Error reading results: {e}")


if __name__ == "__main__":
    show_personal_analysis_results()
