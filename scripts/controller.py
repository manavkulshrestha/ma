import numpy as np
from scipy.spatial.transform import Rotation as R

from utility import dist, signed_rad


class Controller:
    def __init__(self, name, *, target, data, kp=1.3, ki=0, kd=0.9, ksp=1, ksd=100):
        # gain settings
        self.kp = kp
        self.ki = ki#0.01
        self.kd = kd

        self.ksp = ksp
        # self.ksi = 0
        self.ksd = ksd

        # robot pose
        self.target = np.array(target)
        self.get_pos = lambda : data.body(name).xpos
        self.get_orn3D = lambda : data.body(name).xquat
        self.get_orn2D = lambda : signed_rad(np.pi-R.from_quat(data.body(name).xquat).as_euler('xyz')[0])

        # error tracking
        self.e_prev = 0
        self.e_total = 0

        self.ae_prev = 0
        # self.ae_total = 0

        self.e_curr = 0

    def reset(self, target=None):
        if target:
            self.target = np.array(target)

        self.e_prev = 0
        self.e_total = 0

    def get_control(self, scale=0.01, thresh=0.01):
        ''' pid for speed, pursuit for heading '''
        # get distance from target, add for integral
        e_curr, e_vec = dist(self.target, self.get_pos(), get_vec=True)
        self.e_total += e_curr

        # debugging
        self.e_curr = e_curr

        if e_curr < thresh:
            return 0, 0

        # calculate control signal for distance, save error for derivitive
        ctrl_dist = self.kp*e_curr + self.ki*self.e_total + self.kd*(e_curr-self.e_prev)
        self.e_prev = e_curr

        # calculate target angle
        target_angle = np.arctan2(*e_vec[-2::-1])
        angle_error = target_angle - self.get_orn2D()

        ctrl_angle = self.ksp*angle_error + self.ksd*(angle_error - self.ae_prev)
        self.ae_prev = angle_error

        print(f'ANGLE T={target_angle/np.pi:.3f}pi C={self.get_orn2D()/np.pi:.3f}pi', end='')

        # ctrl_dist = 0
        if angle_error > np.pi/3:
            ctrl_dist = 0

        # get wheel torque signals
        left = (ctrl_dist - ctrl_angle)*scale
        right = (ctrl_dist + ctrl_angle)*scale

        return left, right