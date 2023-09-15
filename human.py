import numpy as np
from numpy.linalg import norm

from utility import unit, robot_pos

HERD_RADIUS = 0.5


class Human:
    """
    Currently, simple implementation that moves away from robot if within HERD_RADIUS, else moves randomly

    TODO: add more behaviors: grouped agents, inverse square law repulsion, target based wandering
    """
    def __init__(self, name, *, speed=0.01, robots_num, data, model):
        self.name = name
        self.speed = speed
        self.robots_list = [f'robot{i}' for i in range(robots_num)]

        self.data = data
        self.model = model
        self.done = False

        self.get_pos2D = lambda : model.site(name).pos[:-1]

    def wander(self, target=None, thresh=0.001):
        # pos is a 2d vec, currently. adjustable
        pos = self.get_pos2D()

        gp0, gp1 = np.array(self.model.site('goal0').pos[:2]), np.array(self.model.site('goal1').pos[:2])
        if norm(pos-gp0) < 0.3 or norm(pos-gp1) < 0.3:
            self.done = True
            self.model.site(self.name).pos[-1] = 10
        if self.done:
            return
        
        robots_pos = np.array([robot_pos(name, self.model, dim=2) for name in self.robots_list])
        away_vecs = pos - robots_pos

        displace_vec = np.array([0, 0], dtype=np.float32)
        close_robots = norm(away_vecs, axis=-1) < HERD_RADIUS

        if close_robots.any():
            for away_vec in away_vecs[close_robots]:
                displace_vec += self.speed * unit(away_vec)
        else:
            angle = np.random.uniform(-np.pi, np.pi)
            displace_vec = self.speed * np.array([np.cos(angle), np.sin(angle)])
        
        new_pos = pos + displace_vec
        self.model.site(self.name).pos[:2] = new_pos
        








        
