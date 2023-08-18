import numpy as np


def dist(p1, p2):
    assert type(p1) == type(p2) == np.ndarray

    return np.linalg.norm(p1-p2)

def unit(vec):
    assert type(vec) == np.ndarray

    return vec/np.linalg.norm(vec)