from ripser import ripser
import numpy as np


def clean_jet(jet, max_particles=40):

    mask = jet[:,0] > 0  # ptrel > 0 
    jet = jet[mask]

    jet = jet[np.argsort(jet[:,0])[::-1]]  # most inportant constituents

    return jet[:max_particles]


def compute_diagrams(jet):

    jet = clean_jet(jet)

    if len(jet) < 2:
        return np.empty((0,2)), np.empty((0,2))

    diagrams = ripser(jet, maxdim=1)['dgms']

    H0 = diagrams[0]
    H1 = diagrams[1]

    return H0, H1

