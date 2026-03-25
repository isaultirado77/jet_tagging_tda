import h5py
import numpy as np
from persim import PersistenceImager


def get_diagram(values, offsets, i):

    start = offsets[i]
    end = offsets[i+1]

    return values[start:end].reshape(-1,2)


def clean_diagram(D):
    mask = (
        np.isfinite(D[:, 1]) &  
        ~np.isnan(D).any(axis=1) &  
        (D[:, 1] > D[:, 0])
    )
    return D[mask]


def compute_global_ranges(files):
    bmin, bmax = np.inf, -np.inf
    pmin, pmax = np.inf, -np.inf

    for file in files:
        with h5py.File(file) as f:
            for key_vals, key_off in [("H0_values", "H0_offsets"), ("H1_values", "H1_offsets")]:
                
                vals = f[key_vals][:]
                offs = f[key_off][:]

                for i in range(len(offs) - 1):
                    D = get_diagram(vals, offs, i)
                    D = clean_diagram(D)

                    if len(D) == 0:
                        continue

                    births = D[:, 0]
                    pers = D[:, 1] - births

                    bmin = min(bmin, births.min())
                    bmax = max(bmax, births.max())
                    pmin = min(pmin, pers.min())
                    pmax = max(pmax, pers.max())

    l = max(bmax - bmin, pmax - pmin)
    if l == 0:
        l = 1e-6

    return (bmin, bmin + l), (pmin, pmin + l)


def build_global_imager(files, resolution=40):
    birth_range, pers_range = compute_global_ranges(files)

    pixel_size = (pers_range[1] - pers_range[0]) / resolution

    return PersistenceImager(
        birth_range=birth_range,
        pers_range=pers_range,
        pixel_size=pixel_size,
        kernel_params={'sigma': 0.01}
    )
