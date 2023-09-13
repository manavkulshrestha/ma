import numpy as np



class PointCloud:
    def __init__(self, shape) -> None:
        self.shape = shape



    def get_points(self, depth_frame, camera_pos, z_rot):
        """
        Returns a list of points in the world frame from a depth frame and camera pose.

        Parameters
        ----------
        depth_frame : The depth frame from the camera.
        camera_pos : The position of the camera in the world frame.
        camera_quat : The quaternion of the camera in the world frame.
        """

        size = self.shape

        downsample = 10

        mid_h = size[0]/2
        mid_w = size[1]/2

        index_y = np.linspace(0, size[0]-1, size[0]//downsample, dtype=int)
        index_x = np.linspace(0, size[1]-1, size[1]//downsample, dtype=int)

        points = []

        # Finding [x,y,z] coordinates of each pixel
        for i in range(index_y.shape[0]-1):
            for j in range(index_x.shape[0]-1):

                i_l = index_y[i]
                j_l = index_x[j]

                # Finding the y value based on trigonomety
                D = depth_frame[i_l][j_l]

                if np.isnan(D):
                    continue

                # Obtaining the data from
                x = D
                y = -(j_l - mid_w) * x / size[1]
                z = -(i_l - mid_h) * x / size[0]

                # Saving the point in the world frame
                point = np.array([x, y, z])

                # Rotating the point to the world frame with pivot at the camera
                # z_rotation = np.array([[np.cos(z_rot), -np.sin(z_rot), 0],
                #                        [np.sin(z_rot),  np.cos(z_rot), 0],
                #                        [0,              0,             1]])
                # point = np.matmul(z_rotation, point)
                point = point + camera_pos

                # Adding the point to the list of points
                points.append(point)

        return points



    def get_voxels(self, camera_frame, camera_pose):
        pass