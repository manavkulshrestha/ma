import numpy as np

# Joint restrictions
# Joint 1 (hip)     : -0.863 to 0.863
# Joint 2 (thigh)   : -0.686 to 4.5
# Joint 3 (calf)    : -2.82 to -0.888

class LegController:

    def __init__(self, l1, l2, l3, theta):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.theta = theta

    def ctrl(self) -> np.ndarray:
        # Control the leg
        return np.array([0, 0, -0.888])