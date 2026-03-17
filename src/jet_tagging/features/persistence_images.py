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


def compute_pi_ranges(H):
    if H.size == 0:
        return (0, 1), (0, 1)
    
    births = H[:, 0]
    pers = H[:, 1] - births
    
    bmin, bmax = births.min(), births.max()
    pmin, pmax = pers.min(), pers.max()
    
    l = max(bmax - bmin, pmax - pmin)
    if l == 0: 
        l = 1e-6
    
    birth_range = (bmin, bmin + l)
    pers_range = (pmin, pmin + l)
    
    return birth_range, pers_range


def make_imager(H, resolution=40):

    pimgr = PersistenceImager(
        pixel_size=1/resolution
    )

    return pimgr


def diagram_to_pi(diagram, pimgr):

    diagram = clean_diagram(diagram)

    return pimgr.transform(diagram)
