"""Test cases for the car scraper package."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from car_scraper.models import CarListing
from car_scraper.utils import clean_price, clean_mileage, normalize_fuel_type
from car_scraper.scrape import extract_listing_details


class TestCarListing:
    """Test the CarListing data model."""

    def test_car_listing_creation(self):
        """Test creating a CarListing instance."""
        listing = CarListing(
            detail_url="https://example.com/car/123",
            price=25000,
            kilometers=50000,
            month_of_registration=6,
            year_of_registration=2020,
            horsepower=150,
            fuel_type="Gasoline",
        )

        assert listing.price == 25000
        assert listing.fuel_type == "Gasoline"

    def test_age_calculation(self):
        """Test car age calculation."""
        listing = CarListing(
            detail_url="https://example.com/car/123",
            price=25000,
            kilometers=50000,
            month_of_registration=1,
            year_of_registration=2020,
            horsepower=150,
            fuel_type="Gasoline",
        )

        # Age calculation depends on current date, so just check it's positive
        assert listing.age_in_years > 0

    def test_price_per_hp(self):
        """Test price per horsepower calculation."""
        listing = CarListing(
            detail_url="https://example.com/car/123",
            price=30000,
            kilometers=50000,
            month_of_registration=1,
            year_of_registration=2020,
            horsepower=150,
            fuel_type="Gasoline",
        )

        assert listing.price_per_hp == 200.0  # 30000 / 150


class TestUtils:
    """Test utility functions."""

    def test_clean_price(self):
        """Test price cleaning function."""
        assert clean_price("€ 25.000") == 25000
        assert clean_price("€25,500") == 25500
        assert clean_price("30000") == 30000
        assert clean_price("invalid") is None

    def test_clean_mileage(self):
        """Test mileage cleaning function."""
        assert clean_mileage("50.000 km") == 50000
        assert clean_mileage("75,500km") == 75500
        assert clean_mileage("invalid") is None

    def test_normalize_fuel_type(self):
        """Test fuel type normalization."""
        assert normalize_fuel_type("gasoline") == "Gasoline"
        assert normalize_fuel_type("DIESEL") == "Diesel"
        assert normalize_fuel_type("electric") == "Electric"


class TestScraper:
    """Test scraping functions."""

    @pytest.mark.asyncio
    async def test_extract_listing_details(self):
        """Test detail extraction from listing page."""
        # Mock page object
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=None)
        mock_page.inner_text = AsyncMock(return_value="Gearbox Manual Colour Red")

        result = await extract_listing_details(mock_page, "https://example.com")

        assert result is not None
        assert isinstance(result, dict)
        mock_page.goto.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
