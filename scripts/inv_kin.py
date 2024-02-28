import numpy as np

class inv_kin:

    def __init__(self,
                 l1,
                 l2,
                 l3,
                 phi) -> None:
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.phi = phi
        pass

    def single_leg_ik(self, x, y, z, a1) -> list:

        # YZ-Plane

        # Finding length from leg origin to target
        d = np.linalg.norm([x,y])

        # Finding angle relative to target point (using sine rule)
        a2 = np.arcsin(self.l1/d * np.sin(self.phi))

        # Finding angle relative to origin
        a3 = np.pi - a2 - self.phi

        # Finding the angle respect to the XY-plane
        theta1 = a1 - a3


        # XZ-Plane 