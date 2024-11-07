# config.py
import os
from pathlib import Path
import logging

class Config:
    """Configuration settings for the application"""
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    CLEANED_DIR = DATA_DIR / "cleaned"
    ATTRIBUTED_DATA_DIR = CLEANED_DIR / "attributed"

    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    CLEANED_DIR.mkdir(exist_ok=True)
    ATTRIBUTED_DATA_DIR.mkdir(exist_ok=True)

    # File paths
    RAW_DATA_FILE = UPLOADS_DIR / "martech.csv"
    GENERAL_DATA_FILE = CLEANED_DIR / "general_data.csv"
    ATTRIBUTED_DATA_FILE = ATTRIBUTED_DATA_DIR

    # API settings
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    API_URL = f"http://{API_HOST}:{API_PORT}"
    API_TITLE = "Partner Analysis API"
    API_VERSION = "1.0.0"

    # Cache settings
    CACHE_EXPIRY = 3600

    # CORS settings
    CORS_ORIGINS = ["*"]

    # Logging settings
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = logging.INFO
    LOG_FILE = BASE_DIR / "app.log"

    # Sample data
    SAMPLE_DATA = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Partner': ['Partner A', 'Partner A', 'Partner A'],
        'Leads': [100, 120, 110],
        'Spend': [1000, 1200, 1100],
        'Goals': [20, 24, 22],
        'Locations': ['Location1', 'Location2', 'Location1'],
        'Primary': ['Target1', 'Target2', 'Target1'],
        'Secondary': ['Secondary1', 'Secondary2', 'Secondary1'],
        'Ad': ['Ad1', 'Ad2', 'Ad1']
    }

    # Required columns
    REQUIRED_COLUMNS = ['Date', 'Partner', 'Leads', 'Spend', 'Goals']

    # Attribution columns
    ATTRIBUTION_COLUMNS = ['Locations', 'Primary', 'Secondary']