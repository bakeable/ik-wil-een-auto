"""Data models for car listings."""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class CarListing:
    """Data model for a car listing."""

    detail_url: str
    price: int
    kilometers: int
    month_of_registration: int
    year_of_registration: int
    horsepower: int
    fuel_type: str
    gearbox: Optional[str] = None
    emission_class: Optional[str] = None
    co2_emissions: Optional[str] = None
    colour: Optional[str] = None
    manufacturer_colour: Optional[str] = None
    paint: Optional[str] = None
    upholstery_colour: Optional[str] = None
    upholstery: Optional[str] = None

    @property
    def age_in_years(self) -> float:
        """Calculate car age in years from registration date."""
        current_year = datetime.now().year
        current_month = datetime.now().month

        age_years = current_year - self.year_of_registration
        age_months = current_month - self.month_of_registration

        return age_years + (age_months / 12)

    @property
    def price_per_hp(self) -> float:
        """Calculate price per horsepower ratio."""
        if self.horsepower > 0:
            return self.price / self.horsepower
        return 0.0

    @property
    def kilometers_per_year(self) -> float:
        """Calculate average kilometers per year."""
        age = self.age_in_years
        if age > 0:
            return self.kilometers / age
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "detail_url": self.detail_url,
            "price": self.price,
            "kilometers": self.kilometers,
            "month_of_registration": self.month_of_registration,
            "year_of_registration": self.year_of_registration,
            "horsepower": self.horsepower,
            "fuel_type": self.fuel_type,
            "gearbox": self.gearbox,
            "emission_class": self.emission_class,
            "co2_emissions": self.co2_emissions,
            "colour": self.colour,
            "manufacturer_colour": self.manufacturer_colour,
            "paint": self.paint,
            "upholstery_colour": self.upholstery_colour,
            "upholstery": self.upholstery,
            "age_in_years": round(self.age_in_years, 2),
            "price_per_hp": round(self.price_per_hp, 2),
            "kilometers_per_year": round(self.kilometers_per_year, 0),
        }
