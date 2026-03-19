import json
from pathlib import Path
from jet_tagging.config import RESULTS_DIR


def create_run_dir(mode): 
    base = RESULTS_DIR / mode
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run_*"))

    run_id = len(existing) + 1
    run_dir = base / f"run_{run_id:03d}"
    
    run_dir.mkdir()

    return run_dir


def save_json(data, path): 
    with open(path, 'w') as f: 
        json.dump(data, f, indent=4)
