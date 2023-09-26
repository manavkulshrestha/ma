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

def body_pos(name, access, dim=3):
    return access.body(name).xpos[:dim]

def uniform_displacevec(constant):
    angle = np.random.uniform(-np.pi, np.pi)
    displace_vec = constant * np.array([np.cos(angle), np.sin(angle)])

    return (displace_vec, angle) if return_angle else displace_vec