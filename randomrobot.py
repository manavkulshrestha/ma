import numpy as np
from numpy.linalg import norm

from utility import unit, robot_pos

HERD_RADIUS = 0.5


class Robot:
    def __init__(self, name, *, speed=0.03, data, model):
        self.name = name
        self.speed = speed

        self.data = data
        self.model = model

        self.get_pos2D = lambda : model.site(name).pos[:-1]

    def herd(self, target=None, thresh=0.001):
        # pos is a 2d vec, currently. adjustable
        pos = self.get_pos2D()
        angle = np.random.uniform(-np.pi, np.pi)
        displace_vec = self.speed * np.array([np.cos(angle), np.sin(angle)])
        
        new_pos = pos + displace_vec
        self.model.site(self.name).pos[:2] = new_pos
        








        
