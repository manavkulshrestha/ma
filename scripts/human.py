import numpy as np

from utility import unit

HERD_RADIUS = 0.05


class Human:
    """
    Currently, simple implementation that moves away from robot if within HERD_RADIUS, else moves randomly

    TODO: add more behaviors: grouped agents, inverse square law repulsion, target based wandering
    """
    def __init__(self, name, *, speed=0.05, model):
        self.name = name
        self.speed = speed

        self.model = model

        self.get_pos2D = lambda : model.site(name).pos[:-1]

    def wander(self, target=None, thresh=0.001):
        pos = self.get_pos2D()
        # away_vec = pos - target

        # if np.linalg.norm(away_vec) < HERD_RADIUS:
        #     displace_vec = self.speed * unit(away_vec)
        # else:
        angle = np.random.uniform(-np.pi, np.pi)
        displace_vec = self.speed * np.array([np.cos(angle), np.sin(angle)])

        new_pos = pos + displace_vec
        self.model.site(self.name).pos[:2] = new_pos
        








        
