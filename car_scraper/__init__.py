"""Car Scraper Package - Main module for web scraping and data analysis."""

__version__ = "0.1.0"
__author__ = "Robin Bakker"

from .scrape import scrape_autoscout, extract_listing_details
from .models import CarListing
from .utils import clean_price, clean_mileage, parse_date

__all__ = [
    "scrape_autoscout",
    "extract_listing_details",
    "CarListing",
    "clean_price",
    "clean_mileage",
    "parse_date",
]
