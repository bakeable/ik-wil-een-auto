# Car Sales Data Scraper and Analyzer

A comprehensive Python project for scraping car sales data from AutoScout24 and performing statistical analysis to determine the best value cars.

## Features

- **Web Scraping**: Automated data collection from AutoScout24 using Playwright
- **Data Processing**: Clean and structure car listing data
- **Statistical Analysis**: Advanced modeling to identify best value vehicles
- **Visualization**: Interactive charts and plots for data exploration
- **Export**: Save results to CSV, Excel, and other formats

## Project Structure

```
ik-wil-een-auto/
├── car_scraper/           # Main package
│   ├── scrape.py         # Web scraping logic
│   ├── analysis.py       # Statistical analysis
│   ├── models.py         # Data models
│   └── utils.py          # Utility functions
├── data/                 # Data storage
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit tests
└── reports/              # Generated reports
```

## Installation

1. **Install Poetry** (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Install dependencies**:

```bash
poetry install
```

3. **Install Playwright browsers**:

```bash
poetry run playwright install chromium
```

## Usage

### Scraping Car Data

```bash
# Run the scraper
poetry run scrape-cars

# Or run directly with Python
poetry run python -m car_scraper.scrape
```

### Data Analysis

```bash
# Run analysis
poetry run analyze-cars

# Or use Jupyter notebooks
poetry run jupyter lab notebooks/
```

### Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black car_scraper/
poetry run isort car_scraper/

# Type checking
poetry run mypy car_scraper/
```

## Data Collection

The scraper collects the following car attributes:

- Price and registration details
- Engine specifications (horsepower, fuel type)
- Mileage and condition
- Technical specifications (gearbox, emissions)
- Aesthetic details (color, upholstery)

## Analysis Features

- Price prediction modeling
- Value-for-money scoring
- Market trend analysis
- Depreciation calculations
- Feature importance analysis

## Requirements

- Python 3.9+
- Poetry for dependency management
- Chromium browser (installed via Playwright)

## License

MIT License
