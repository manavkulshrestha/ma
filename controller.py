import numpy as np
from utility import dist

class Controller:
    def __init__(self, name, *, target, data):
        # gain settings
        self.kp = 1
        self.ki = 0.01
        self.kd = 0.5

        # robot pose
        self.target = target
        self.get_pos = lambda : data.body(name).xpos
        self.get_orn = lambda : data.body(name).xquat # TODO project to 2D? maybe not necessary

        # error tracking
        self.e_prev = 0
        self.e_total = 0


    def get_control(self):
        # pid for speed, pursuit for heading
        e_curr = self.dist(self.target, self.get_pos())
        self.e_total += e_curr

        ctrl_dist = self.kp*e_curr + self.ki*self.e_total + self.kd*(e_curr-self.e_prev)
        self.e_prev = e_curr

        
        ctrl_angle = np.arctan2(*e_curr[-2::-1])

        left = ctrl_dist + ctrl_angle
        right = -ctrl_dist + ctrl_angle

        return left, right





