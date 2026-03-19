from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Paths
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
JETS_RAW_DIR = RAW_DIR / "jets"
PROCESSED_DIR = DATA_DIR / "processed"
JETS_PROCESSED_DIR = PROCESSED_DIR / "jets"
TOPOLOGY_DIR = DATA_DIR / "topology"
MERGED_DATASETS_DIR = DATA_DIR / "merged"


# helper functions
def ensure_directories():
    dirs = [
        DATA_DIR, 
        RAW_DIR, 
        PROCESSED_DIR, 
        JETS_RAW_DIR, 
        JETS_PROCESSED_DIR, 
        TOPOLOGY_DIR
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
