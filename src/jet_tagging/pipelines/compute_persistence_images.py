from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from persim import PersistenceImager

from jet_tagging.config import (
    DATA_DIR,
    TOPOLOGY_DIR, 
)
from jet_tagging.features.persistence_images import (
    get_diagram, 
    clean_diagram
)


def process_file(input_path, output_path, resolution=40):

    with h5py.File(input_path) as f:

        H0_vals = f["H0_values"][:]
        H0_off = f["H0_offsets"][:]

        H1_vals = f["H1_values"][:]
        H1_off = f["H1_offsets"][:]

        labels = f["labels"][:]

    n = len(labels)

    pimgr = PersistenceImager(pixel_size=1/resolution)

    imgs_H0 = []
    imgs_H1 = []
    imgs_S = []  # sum of persistence images


    for i in tqdm(range(n)):

        H0 = get_diagram(H0_vals, H0_off, i)
        H1 = get_diagram(H1_vals, H1_off, i)

        H0 = clean_diagram(H0)
        H1 = clean_diagram(H1)

        img0 = pimgr.transform(H0)
        img1 = pimgr.transform(H1)

        imgS = img0 + img1

        imgs_H0.append(img0)
        imgs_H1.append(img1)
        imgs_S.append(imgS)

    imgs_H0 = np.array(imgs_H0)
    imgs_H1 = np.array(imgs_H1)
    imgs_S = np.array(imgs_S)


    with h5py.File(output_path, 'w') as f:

        f.create_dataset(
            "pi_H0",
            data=imgs_H0,
            compression='gzip'
        )

        f.create_dataset(
            "pi_H1",
            data=imgs_H1,
            compression='gzip'
        )

        f.create_dataset(
            "pi_S",
            data=imgs_S,
            compression='gzip'
        )

        f.create_dataset(
            "labels",
            data=labels
        )


def compute_persistence_images_dataset(input_dir, output_dir):

    files = sorted(Path(input_dir).glob("*.h5"))

    output_dir.mkdir(parents=True, exist_ok=True)

    for file in files:

        out = output_dir / f"pi_{file.stem}.h5"

        if out.exists():
            continue

        process_file(file, out)


def main(): 
    dgms_dir = TOPOLOGY_DIR / "diagrams"
    pimgs_dir = TOPOLOGY_DIR / "persistence_images"
    compute_persistence_images_dataset(dgms_dir, pimgs_dir)


if __name__ == '__main__': 
    main()
