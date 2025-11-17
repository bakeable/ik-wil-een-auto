"""Web scraping module for AutoScout24 car listings."""

import asyncio
import csv
import hashlib
import random
import re
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Set

from playwright.async_api import async_playwright, Page

from .models import CarListing
from .utils import clean_price, clean_mileage, normalize_fuel_type


# Regex patterns for data extraction
PRICE_RE = re.compile(r"€\s?([\d.,]+)")
KM_RE = re.compile(r"([\d.,]+)\s*km")
DATE_RE = re.compile(r"(\d{2})/(\d{4})")
FUEL_RE = re.compile(
    r"(Electric/Gasoline|Hybrid\s*\(Electric/Gasoline\)|Hybrid\s*\(Electric/Diesel\)|Gasoline|Diesel|Electric|CNG|LPG|Hydrogen|Ethanol)",
    re.IGNORECASE,
)
HP_RE = re.compile(r"\((\d+)\s*hp\)")
POWER_KW_HP_RE = re.compile(r"(\d+)\s*kW\s*\((\d+)\s*hp\)")

# Additional detail patterns - made more precise to avoid capturing too much text
GEARBOX_RE = re.compile(r"Gearbox\s+([A-Za-z]+)(?:\s|$|\n)", re.IGNORECASE)
EMISSION_RE = re.compile(
    r"Emission class\s+(Euro\s*[\w\s.-]+?)(?:\s*\n|\s*Fuel|\s*CO₂|\s*$)", re.IGNORECASE
)
CO2_RE = re.compile(r"CO₂-emissions\s*([\d.,]+)\s*g/km", re.IGNORECASE)
COLOUR_RE = re.compile(
    r"(?:^|\n)Colour\s+([\w\s]+?)(?:\s*\n|\s*Manufacturer|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
MANUF_COLOUR_RE = re.compile(
    r"Manufacturer colour\s+([\w\s]+?)(?:\s*\n|\s*Paint|\s*$)", re.IGNORECASE
)
PAINT_RE = re.compile(r"Paint\s+([\w\s]+?)(?:\s*\n|\s*Upholstery|\s*$)", re.IGNORECASE)
UPH_COLOUR_RE = re.compile(
    r"Upholstery colour\s+([\w\s]+?)(?:\s*\n|\s*Upholstery\s|\s*$)", re.IGNORECASE
)
UPHOLSTERY_RE = re.compile(
    r"(?:^|\n)Upholstery\s+([\w\s]+?)(?:\s*\n|\s*Vehicle|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


async def random_human_delay(min_seconds: float = 0, max_seconds: float = 2):
    """Add random delay to simulate human behavior."""
    delay = random.uniform(0, 1)
    await asyncio.sleep(delay)


async def random_scroll(page: Page):
    """Perform random scrolling to simulate human behavior."""
    # Get page dimensions
    viewport = page.viewport_size
    if not viewport:
        return

    # Random scroll patterns
    scroll_patterns = [
        # Slow scroll down
        lambda: page.mouse.wheel(0, random.randint(200, 800)),
        # Quick scroll down
        lambda: page.mouse.wheel(0, random.randint(800, 1500)),
        # Scroll up a bit
        lambda: page.mouse.wheel(0, random.randint(-300, -100)),
        # Small scroll movements
        lambda: page.mouse.wheel(0, random.randint(50, 200)),
    ]

    # Perform 2-4 random scroll actions
    for _ in range(random.randint(2, 4)):
        scroll_action = random.choice(scroll_patterns)
        await scroll_action()
        await random_human_delay(0.3, 1.2)


async def random_mouse_movement(page: Page):
    """Simulate random mouse movements."""
    viewport = page.viewport_size
    if not viewport:
        return

    # Move mouse to random positions
    for _ in range(random.randint(2, 5)):
        x = random.randint(50, viewport["width"] - 50)
        y = random.randint(50, viewport["height"] - 50)
        await page.mouse.move(x, y)
        await random_human_delay(0.2, 0.8)


async def simulate_human_page_interaction(page: Page):
    """Simulate human-like page interaction before scraping."""
    # Random delay before starting
    await random_human_delay(0, 1.5)

    # Random mouse movement
    await random_mouse_movement(page)

    # Random scrolling
    await random_scroll(page)

    # Another random delay
    await random_human_delay(0, 1.5)


def generate_listing_hash(detail_url: str) -> str:
    """Generate a unique hash for a listing based on its URL."""
    return hashlib.md5(detail_url.encode()).hexdigest()


def load_existing_hashes(output_file: Path) -> Set[str]:
    """Load existing hashes from CSV file to avoid duplicates."""
    existing_hashes = set()
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "listing_hash" in row and row["listing_hash"]:
                        existing_hashes.add(row["listing_hash"])
        except Exception as e:
            print(f"Warning: Could not load existing hashes: {e}")
    return existing_hashes


def append_to_csv(listing: CarListing, output_file: Path, write_header: bool = False):
    """Append a single listing to CSV file immediately."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "listing_hash",
        "detail_url",
        "price",
        "kilometers",
        "month_of_registration",
        "year_of_registration",
        "horsepower",
        "fuel_type",
        "gearbox",
        "emission_class",
        "co2_emissions",
        "colour",
        "manufacturer_colour",
        "paint",
        "upholstery_colour",
        "upholstery",
        "age_in_years",
        "price_per_hp",
        "kilometers_per_year",
    ]

    mode = "w" if write_header else "a"
    with output_file.open(mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # Add hash to the listing data
        listing_data = listing.to_dict()
        listing_data["listing_hash"] = generate_listing_hash(listing.detail_url)
        writer.writerow(listing_data)


async def extract_listing_details(page: Page, url: str) -> Dict[str, Optional[str]]:
    """Extract detailed information from a car listing page.

    Args:
        page: Playwright page instance
        url: URL of the car listing

    Returns:
        Dictionary containing extracted car details
    """
    await page.goto(url, timeout=120_000)

    # Simulate human-like behavior on detail page
    await random_human_delay(1, 2)
    await simulate_human_page_interaction(page)

    try:
        # Handle cookie acceptance popup
        accept_btn = await page.query_selector("button:has-text('Accept')")
        if accept_btn:
            await accept_btn.click()
            await random_human_delay(0.5, 1)
    except Exception:
        pass

    # Random delay before extracting content
    await random_human_delay(1, 2)

    body_text = await page.inner_text("body")

    def search(regex: re.Pattern) -> Optional[str]:
        """Search for a pattern in the body text and clean the result."""
        match = regex.search(body_text)
        if match:
            result = match.group(1).strip()
            # Limit length and clean up common issues
            if len(result) > 50:  # If result is too long, likely captured too much
                result = result.split("\n")[0].strip()  # Take only first line
                if len(result) > 50:
                    result = result[:50].strip()
            return result if result else None
        return None

    return {
        "gearbox": search(GEARBOX_RE),
        "emission_class": search(EMISSION_RE),
        "co2_emissions": search(CO2_RE),
        "colour": search(COLOUR_RE),
        "manufacturer_colour": search(MANUF_COLOUR_RE),
        "paint": search(PAINT_RE),
        "upholstery_colour": search(UPH_COLOUR_RE),
        "upholstery": search(UPHOLSTERY_RE),
    }


async def scrape_autoscout(
    make: str = "peugeot",
    model: str = "2008",
    max_pages: int = 65,
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> List[CarListing]:
    """Scrape car listings from AutoScout24.

    Args:
        make: Car manufacturer (default: peugeot)
        model: Car model (default: 2008)
        max_pages: Maximum number of pages to scrape
        output_file: Path to save CSV file (optional)
        verbose: Print progress messages

    Returns:
        List of CarListing objects
    """
    async with async_playwright() as playwright:
        # Launch browser with realistic settings to avoid detection
        browser = await playwright.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--no-first-run",
                "--disable-default-apps",
                "--disable-extensions",
                "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ],
        )

        # Create context with realistic settings
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="Europe/Amsterdam",
        )

        list_page = await context.new_page()
        detail_page = await context.new_page()

        base_url_template = f"https://www.autoscout24.com/lst/{make}/{model}?atype=C&cy=NL&damaged_listing=exclude&desc=0&page={{page}}&search_id=yon2xdqqgm&sort=standard&source=listpage_pagination&ustate=N%2CU"

        results: List[CarListing] = []

        # Set up output file and load existing hashes
        output_path = (
            Path(output_file) if output_file else Path("data/car_listings.csv")
        )
        existing_hashes = load_existing_hashes(output_path)
        write_header = not output_path.exists() or output_path.stat().st_size == 0

        if verbose:
            print(
                f"Found {len(existing_hashes)} existing listings, will skip duplicates"
            )
            print(f"Saving results incrementally to: {output_path}")

        for page_number in range(1, max_pages + 1):
            if verbose:
                print(f"Processing list page {page_number}/{max_pages} ...")

            list_url = base_url_template.format(page=page_number)
            await list_page.goto(list_url, timeout=120_000)

            # Ensure DOM is fully loaded
            await list_page.wait_for_load_state("domcontentloaded")

            # Simulate human-like page interaction
            await simulate_human_page_interaction(list_page)

            try:
                # Handle cookie acceptance popup on each page
                accept_btn = await list_page.query_selector("button:has-text('Accept')")
                if accept_btn:
                    await accept_btn.click()
                    await random_human_delay(1, 2)
            except Exception:
                pass

            # Random delay before looking for elements
            await random_human_delay(0, 3)

            await list_page.wait_for_selector(
                "article[data-testid='list-item']", timeout=30_000
            )
            items = await list_page.query_selector_all(
                "article[data-testid='list-item']"
            )

            # Random delay after getting items
            await random_human_delay(0, 3)

            for i, item in enumerate(items):
                # Random delay between processing items
                if i > 0:
                    await random_human_delay(0.5, 2)

                # Occasionally simulate mouse hover over item
                if random.random() < 0.3:
                    try:
                        await item.hover()
                        await random_human_delay(0.3, 0.8)
                    except Exception:
                        pass

                item_text = (await item.inner_text()).replace("\n", " ")

                # Extract basic information using regex
                price_match = PRICE_RE.search(item_text)
                km_match = KM_RE.search(item_text)
                date_match = DATE_RE.search(item_text)
                fuel_match = FUEL_RE.search(item_text)
                hp_match = HP_RE.search(item_text)

                if not (
                    price_match and km_match and date_match and fuel_match and hp_match
                ):
                    continue

                # Parse extracted data
                price = clean_price(price_match.group(1))
                kilometers = clean_mileage(km_match.group(1))
                fuel_type = normalize_fuel_type(fuel_match.group(1))

                # Skip if any required fields are None
                if price is None or kilometers is None or fuel_type is None:
                    continue

                month = int(date_match.group(1))
                year = int(date_match.group(2))
                horsepower = int(hp_match.group(1))

                # Get detail page URL
                link_elem = await item.query_selector(
                    "a[href^='https://www.autoscout24.com/offers/']"
                )
                if not link_elem:
                    continue
                detail_url = await link_elem.get_attribute("href")

                # Check for duplicates using hash
                listing_hash = generate_listing_hash(detail_url)
                if listing_hash in existing_hashes:
                    if verbose:
                        print(f"  Skipping duplicate listing: {detail_url[:60]}...")
                    continue

                # Extract additional details from the car's detail page
                await random_human_delay(1, 3)  # Random delay before detail page
                extended = await extract_listing_details(detail_page, detail_url)

                # Random delay after detail extraction
                await random_human_delay(0.5, 1.5)

                # Create CarListing object
                car_listing = CarListing(
                    detail_url=detail_url,
                    price=price,
                    kilometers=kilometers,
                    month_of_registration=month,
                    year_of_registration=year,
                    horsepower=horsepower,
                    fuel_type=fuel_type,
                    gearbox=extended["gearbox"],
                    emission_class=extended["emission_class"],
                    co2_emissions=extended["co2_emissions"],
                    colour=extended["colour"],
                    manufacturer_colour=extended["manufacturer_colour"],
                    paint=extended["paint"],
                    upholstery_colour=extended["upholstery_colour"],
                    upholstery=extended["upholstery"],
                )

                # Save immediately to CSV
                try:
                    append_to_csv(car_listing, output_path, write_header)
                    write_header = False  # Only write header once
                    existing_hashes.add(listing_hash)  # Track this hash
                    results.append(car_listing)

                    if verbose:
                        print(
                            f"  Saved listing #{len(results)}: €{price:,} - {detail_url[:60]}..."
                        )
                        print(
                            f"    Price: €{price:,}, Age: {car_listing.age_in_years:.1f} years, {kilometers:,} km, {horsepower} hp, Fuel: {fuel_type}"
                        )

                except Exception as e:
                    print(f"  Warning: Failed to save listing: {e}")

            if verbose:
                print(
                    f"Page {page_number} complete. Total saved: {len(results)} listings"
                )

            # Random delay between pages (longer for realistic browsing)
            if page_number < max_pages:
                await random_human_delay(0, 3)

        await browser.close()

        if verbose:
            print(f"\nScraping completed! Total listings saved: {len(results)}")
            print(f"Data saved to: {output_path.resolve()}")

        return results


def save_to_csv(listings: List[CarListing], output_path: str) -> None:
    """Save car listings to CSV file.

    Args:
        listings: List of CarListing objects
        output_path: Path to output CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "detail_url",
        "price",
        "kilometers",
        "month_of_registration",
        "year_of_registration",
        "horsepower",
        "fuel_type",
        "gearbox",
        "emission_class",
        "co2_emissions",
        "colour",
        "manufacturer_colour",
        "paint",
        "upholstery_colour",
        "upholstery",
        "age_in_years",
        "price_per_hp",
        "kilometers_per_year",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([listing.to_dict() for listing in listings])


def main() -> None:
    """Main entry point for the scraper."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape car listings from AutoScout24")
    parser.add_argument("--make", default="peugeot", help="Car manufacturer")
    parser.add_argument("--model", default="2008", help="Car model")
    parser.add_argument(
        "--max-pages", type=int, default=100, help="Maximum pages to scrape"
    )
    parser.add_argument(
        "--output", default="data/car_listings.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages"
    )

    args = parser.parse_args()

    async def run_scraper():
        await scrape_autoscout(
            make=args.make,
            model=args.model,
            max_pages=args.max_pages,
            output_file=args.output,
            verbose=not args.quiet,
        )

    asyncio.run(run_scraper())


if __name__ == "__main__":
    main()
