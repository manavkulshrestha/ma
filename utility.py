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
    return zip(*[lst[i:] for i in range(n)])


class ModelManager:
    def __init__(self, cls, name, time_dir='', *, save_every=(), save_best=True):
        if not time_dir:
            self.dir = MODELS_PATH/time_label()
            self.dir.mkdir()
        self.cls = cls
        self.name = name

        self.save_every = (save_every,) if isinstance(save_every, int) else save_every
        self.save_best = save_best

        self.best_score = 0
        # TODO have epoch=best also save epoch number and regex for loading

    def save(self, model, *, epoch):
        torch.save(model.state_dict(), self.dir/f'{self.name}-{epoch}.pt')

    def load(self, epoch, model_args=[], model_kwargs={}, cuda=True):
        model = self.cls(*model_args, **model_kwargs)
        model.load_state_dict(torch.load(self.dir/f'{self.name}-{epoch}.pt'))

        return model.cuda() if cuda else model
    
    def saves(self, model, epoch, score):
        if self.save_best:
            if self.best_score < score:
                self.best_score = score
                self.save(model, epoch='best')
        for save_mult in self.save_every:
            if epoch % save_mult == 0:
                self.save(model, epoch=epoch)
                return