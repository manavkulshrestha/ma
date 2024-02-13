import numpy as np


def dist(p1, p2, get_vec=False):
    assert type(p1) == type(p2) == np.ndarray

    diff = p1-p2
    if get_vec:
        return np.linalg.norm(diff), diff

    return np.linalg.norm(diff)

def unit(vec):
    assert type(vec) == np.ndarray

    return vec/np.linalg.norm(vec)

def signed_rad(rad):
    rad = rad % (2*np.pi)

    if rad > np.pi:
        return rad - 2*np.pi
    
    return rad
