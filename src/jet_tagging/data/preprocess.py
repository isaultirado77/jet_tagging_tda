from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from tqdm import tqdm

from jet_tagging.config import (
    JETS_RAW_DIR, 
    JETS_PROCESSED_DIR
)

JET_FEATURE_IDX = [53, 54]
PARTICLE_FEATURE_IDX = [6, 9, 12]


def process_file(input_path, output_path):

    with h5py.File(input_path, 'r') as f:

        jets = f["jets"][:, JET_FEATURE_IDX]
        particles = f["jetConstituentList"][:, :, PARTICLE_FEATURE_IDX]
        images = f["jetImage"][:]

    mask = np.logical_or(jets[:,0] == 1, jets[:,1] == 1)

    jets = jets[mask][:,0]
    particles = particles[mask]
    images = images[mask]

    with h5py.File(output_path, 'w') as f:

        f.create_dataset(
            "particles",
            data=particles,
            compression='gzip'
        )

        f.create_dataset(
            "images",
            data=images,
            compression='gzip'
        )

        f.create_dataset(
            "labels",
            data=jets
        )


def build_dataset(input_dir, output_dir):

    input_files = sorted(Path(input_dir).glob("*.h5"))

    output_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(input_files, desc="Processing files"):

        out = output_dir / f"processed_{file.stem}.h5"

        if out.exists():
            continue

        process_file(file, out)


def process_task(task): 
    input_path, output_path = task
    return process_file(input_path, output_path)


def build_dataset_parallel(input_dir, output_dir, workers=8):

    input_files = sorted(Path(input_dir).glob("*.h5"))

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    for file in input_files:

        out = output_dir / f"processed_{file.stem}.h5"

        if out.exists():
            continue

        tasks.append((file, out))

    with ProcessPoolExecutor(workers) as executor:

        list(
            tqdm(
                executor.map(process_task, tasks),
                total=len(tasks)
            )
        )


def main(): 
    train_dir = JETS_RAW_DIR / "train"
    build_dataset(train_dir, JETS_PROCESSED_DIR)


if __name__ == "__main__":
    main()