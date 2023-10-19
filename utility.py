from datetime import datetime
from typing import Iterable
import numpy as np
import pickle
import torch
from pathlib import Path


MODELS_PATH = Path('models')


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

def uniform_displacevec(constant, return_angle=False):
    angle = np.random.uniform(-np.pi, np.pi)
    displace_vec = constant * np.array([np.cos(angle), np.sin(angle)])

    return (displace_vec, angle) if return_angle else displace_vec

def time_label():
    return datetime.now().strftime('%y-%m-%d-%H%M%S%f')[:17]

def save_pkl(obj, s, ext=False):
    with open(f'{s}.pkl' if ext else s, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(s, ext=False):
    with open(f'{s}.pkl' if ext else s, 'rb') as f:
        return pickle.load(f)
    
def sliding(lst: Iterable, n: int):
    """ returns a sliding window of size n over a list lst """
    for window in zip(*[lst[i:] for i in range(n)]):
        yield window

def save_model(model, name):
    torch.save(model.state_dict(), MODELS_PATH/name)

def load_model(model_cls, name, model_args=[], model_kwargs={}, cuda=True):
    model = model_cls(*model_args, **model_kwargs)
    model.load_state_dict(torch.load(MODELS_PATH/name))

    return model.cuda() if cuda else model