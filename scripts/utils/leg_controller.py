import numpy as np

# Joint restrictions
# Joint 1 (hip)     : -0.863 to 0.863
# Joint 2 (thigh)   : -0.686 to 4.5
# Joint 3 (calf)    : -2.82 to -0.888

class LegController:

    def __init__(self, l1, l2, l3, phi):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.phi = phi

    def atan(z, x):
        if (x > 0 and z >= 0):
            return np.arctan(z/x)
        elif (x == 0 and z >= 0):
            return np.pi/2
        elif (x < 0 and z >= 0):
            return -abs(np.arctan(z/x) + np.pi)
        elif (x < 0 and z < 0):
            return np.arctan(z/x) + np.pi
        elif (x > 0 and ):
            
        elif (x == 0 and z < 0):
            return -np.pi/2
        else:
            return 0

    def ctrl(self, goal, alpha1) -> np.ndarray:
        # Control the leg

        # Linear distance from fromt (origin to goal)
        a = np.linalg.norm(np.array([goal[0], goal[1]]))
        alpha2 = np.arcsin(self.l1/a * np.sin(self.phi))
        alpha3 = np.pi - alpha2 - self.phi
        theta1 = alpha1 - alpha3

        # Calculate the position of the end effector
        r = theta1 + self.phi - np.pi/2

        b = np.linalg.norm(np.array([goal[0], goal[2]]))
        beta1 = np.arctan(goal[2]/goal[0])


        return np.array([0, 0, -0.888])
    