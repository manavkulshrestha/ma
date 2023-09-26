import time

import numpy as np
from numpy.linalg import norm

from utility import uniform_displacevec, unit, body_pos
from perlin_noise import PerlinNoise

HERD_RADIUS = 0.5

class Agent:
    def __init__(self, name, *, speed, model, data):
        self.name = name
        self.speed = speed

        self._d = data
        self._m = model

        self.travelled = 0
        self.prev_pos = self.pos
        self.initial_pos = self._d.body(self.name).xpos

    @property
    def pos(self, dim=2):
        return self._d.body(self.name).xpos[:dim]
    
    @pos.setter
    def pos(self, new_pos, dim=2):
        self._d.joint(self.name).qpos[:dim] = new_pos[:dim]

    @property
    def vel(self, dim=2):
        return self._d.joint(self.name).qvel[:dim]
    
    @vel.setter
    def vel(self, new_vel, dim=2):
        self._d.joint(self.name).qvel[:dim] = new_vel[:dim]

    @property
    def acc(self, dim=2):
        return self._d.joint(self.name).qacc[:dim]

    @acc.setter
    def acc(self, new_acc, dim=2):
        self._d.joint(self.name).qacc[:dim] = new_acc[:dim]
        
    # orn setter, getter later
    
    def move(self, displace, zero_acc=False, height_fix=True, acc=False):
        # set_pos += displace
        # self.travelled += norm(displace)

        if zero_acc:
            self.acc = np.zeros_like(self.acc)

        if acc:
            self.acc = displace
        else:
            self.vel = displace

        if height_fix:
            self._d.joint(self.name).qpos[2] = self.initial_pos[2]

        self.travelled += norm(self.pos - self.prev_pos)
        self.prev_pos = self.pos

class Robot(Agent):
    def __init__(self, name, *, speed=1, battery=100, model, data):
        super().__init__(name, speed=speed, model=model, data=data)
        self.battery = battery

    def herd(self, target=None, thresh=0.001):
        # pass
        displace_vec, angle = uniform_displacevec(10*self.speed, return_angle=True)
        self.move(displace_vec)
        # self.move(policy())
        return angle


class Human(Agent):
    def __init__(self, name, *, speed=1, robots_num, data, model):
        super().__init__(name, speed=speed, model=model, data=data)
        self.robots_list = [f'robot{i}' for i in range(robots_num)]
        self.done = False

        # self.perlin_x = PerlinNoise()
        # self.perlin_y = PerlinNoise()

    def wander(self, target=None, thresh=0.001):
        pos = self.pos

        # check if goal satisfied
        # gp0, gp1 = np.array(self._m.site('goal0').pos[:2]), np.array(self._m.site('goal1').pos[:2])
        # if norm(pos-gp0) < 0.3 or norm(pos-gp1) < 0.3:
        #     self.done = True
        #     self._m.site(self.name).pos[-1] = 10
        # if self.done:
        #     return
        
        # get all robot positions
        robots_pos = np.array([body_pos(name, self._d, dim=2) for name in self.robots_list])
        away_vecs = pos - robots_pos

        # close robots repell
        displace_vec = np.array([0, 0], dtype=np.float32)
        close_robots = norm(away_vecs, axis=-1) < HERD_RADIUS

        # if any robot is close, repell from them. Else, random movement
        if close_robots.any():
            for away_vec in away_vecs[close_robots]:
                displace_vec += self.speed * away_vec / norm(away_vec)**3
                # self.move(displace_vec)

        else:
            displace_vec = uniform_displacevec(10*self.speed)
            # self.move(displace_vec)
            # displace_vec = np.array([self.perlin_x(time.time()), self.perlin_y(time.time())])

            # pass

        self.move(displace_vec)
        








        









        
