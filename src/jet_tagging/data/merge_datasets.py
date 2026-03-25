from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from jet_tagging.config import (
    DATA_DIR, 
    JETS_PROCESSED_DIR, 
    TOPOLOGY_DIR
)


def merge_datasets(input_dir, output_file, key, total=100000):
    
    files = sorted(Path(input_dir).glob("*.h5"))
    assert len(files) > 0, 'No .h5 files found'

    with h5py.File(files[0], 'r') as f:
        sample_shape = f[key].shape[1:]

    mode = 'a' if Path(output_file).exists() else 'w'

    with h5py.File(output_file, mode) as fout: 
        
        data_out = fout.create_dataset(
            key,
            shape=(total, *sample_shape),
            compression='gzip'
        )

        offset = 0

        for file in tqdm(files):
            if offset >= total:
                break

            with h5py.File(file, 'r') as fin:
                
                if key not in fin:
                    continue

                dset = fin[key]

                n_samples = dset.shape[0]
                remaining = total - offset
                n_to_copy = min(n_samples, remaining)

                data_chunk = dset[:n_to_copy]

                data_out[offset:offset+n_to_copy] = data_chunk
                offset += n_to_copy

        print(f"Merged {offset} samples for key '{key}' into: {output_file}")

def main(): 
    total_samples = 100_000
    topology_keys = [
        'pi_H0', 
        'pi_H1', 
        'labels'
    ]
    jets_keys = [
        'images', 
        'labels',
    ]

    merged_dir = DATA_DIR / "merged"
    merged_dir.mkdir(exist_ok=True)

    pimgs_dir = TOPOLOGY_DIR / "persistence_images"

    pimgs_merged_path = merged_dir / f"persistence_images_merged_{total_samples}.h5"
    jets_merged_path = merged_dir / f"jet_images_merged_{total_samples}.h5"

    for key in topology_keys: 
        _ = merge_datasets(
            pimgs_dir, 
            pimgs_merged_path, 
            key, 
            total_samples
        )

    for key in jets_keys: 
       _ = merge_datasets(
           JETS_PROCESSED_DIR, 
           jets_merged_path, 
           key, 
           total_samples
       )


if __name__ == "__main__": 
    main()
