from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
JETS_DIR = PROCESSED_DIR / "jets"
JET_IMAGES_DIR = PROCESSED_DIR / "images"

# helper functions
def ensure_directories():
    dirs = [
        DATA_DIR, 
        RAW_DIR, 
        PROCESSED_DIR, 
        JETS_DIR, 
        JET_IMAGES_DIR, 
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
