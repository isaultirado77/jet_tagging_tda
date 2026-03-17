from pathlib import Path
import argparse

import h5py
import numpy as np
from tqdm import tqdm

from jet_tagging.config import (
    JETS_PROCESSED_DIR,
    TOPOLOGY_DIR
)

from jet_tagging.features.topology import compute_diagrams


def flatten_diagrams(diagrams):

    values = []
    offsets = [0]

    for d in diagrams:

        values.extend(d.flatten())

        offsets.append(len(values))

    return np.array(values), np.array(offsets)


def process_file(input_path, output_path):

    with h5py.File(input_path, 'r') as f:
        particles = f["particles"][:]
        labels = f["labels"][:]

        n = particles.shape[0]

        H0_all = []
        H1_all = []

        for i in tqdm(range(n)):

            jet = particles[i]

            H0, H1 = compute_diagrams(jet)

            H0_all.append(H0)
            H1_all.append(H1)
    
    H0_vals, H0_off = flatten_diagrams(H0_all)
    H1_vals, H1_off = flatten_diagrams(H1_all)

    with h5py.File(output_path, 'w') as f:

        f.create_dataset(
            "H0_values",
            data=H0_vals,
            compression='gzip'
        )

        f.create_dataset(
            "H0_offsets",
            data=H0_off,
            compression='gzip'
        )

        f.create_dataset(
            "H1_values",
            data=H1_vals,
            compression='gzip'
        )

        f.create_dataset(
            "H1_offsets",
            data=H1_off,
            compression='gzip'
        )

        f.create_dataset(
            "labels",
            data=labels,
        )


def compute_diagrams_dataset(input_dir, output_dir):

    input_files = sorted(Path(input_dir).glob("*.h5"))

    output_dir.mkdir(exist_ok=True, parents=True)

    for file in input_files:

        out = output_dir / f"diagrams_{file.stem}.h5"

        if out.exists():
            continue

        process_file(file, out)
    
    return sorted(output_dir.glob(".h5"))


def main():
    diagrams_dir = TOPOLOGY_DIR / "diagrams"
    _ = compute_diagrams_dataset(JETS_PROCESSED_DIR, diagrams_dir)
if __name__ == '__main__':
    main()

