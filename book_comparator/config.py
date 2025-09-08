import os
from pathlib import Path
from dataclasses import dataclass
import logging
from time import time

@dataclass
class Settings:
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    
    DATA_DIR: Path = Path('data')
    POSITIVE_PAIRS_DIR: Path = DATA_DIR / 'image_pairs_positives'
    NEGATIVE_PAIRS_DIR: Path = DATA_DIR / 'image_pairs_negatives'
    RESULTS_DIR: Path = Path('results')
    
    MODEL = 'gemini-1.5-pro'
    DEFAULT_THRESHOLD: float = 0.35
    DEFAULT_N_SAMPLES: int = 100
    
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_FILE: Path = RESULTS_DIR / f'experiment_{int(time())}.log'

def setup_logging(settings: Settings) -> logging.Logger:
    logger = logging.getLogger('book_comparator')
    logger.setLevel(settings.LOG_LEVEL)
    
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(settings.LOG_FILE)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_settings() -> Settings:
    return Settings()

settings = get_settings()
logger = setup_logging(settings)