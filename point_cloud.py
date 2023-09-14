import numpy as np
import math



class PointCloud:
    def __init__(self, shape) -> None:
        self.shape = shape



    # A custom function to calculate
    # probability distribution function
    def pdf(self, size):
        x = np.arange(0, size, 1)

        mean = np.mean(x)
        std = np.std(x) + 380
        y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
        return y_out



    def get_points(self, depth_frame, camera_pos, z_rot):
        """
        Returns a list of points in the world frame from a depth frame and camera pose.

        Parameters
        ----------
        depth_frame : The depth frame from the camera.
        camera_pos : The position of the camera in the world frame.
        camera_quat : The quaternion of the camera in the world frame.
        """

        height, width = self.shape

        depth_frame = np.expand_dims(depth_frame, axis=2)

        downsample = 5

        mid_h = height/2
        mid_w = width/2

        pdf_w = self.pdf(width)
        max = np.max(pdf_w)
        pdf_w = pdf_w / max

        index_y = np.linspace(0, height-1, height//downsample, dtype=int)
        index_x = np.linspace(0, width-1, width//downsample, dtype=int)

        # Camera intrinsics
        fovy = 60
        c_y = height / 2
        c_x = width / 2
        f_y = 0.5 * height / math.tan(fovy * math.pi / 360)
        f_x = f_y * (width / height)

        points = []

        # Finding [x,y,z] coordinates of each pixel
        for i in index_y:
            for j in index_x:

                D = depth_frame[i][j]

                # Skip if the pixel is not in the image
                if np.isnan(D[0]):
                    continue

                # Obtaining the data from
                # mat = np.array([[1],
                #                 [-(j - mid_w) * scale / width],
                #                 [-(i - mid_h) / height]])
                
                mat = np.array([[(i - c_y) / f_y],
                                [(j - c_x) * (width/height) / f_x],
                                [1]])
                
                # Saving the point in the world frame
                point = np.matmul(mat, D)

                # Rotating the point to the world frame with pivot at the camera
                point = np.matmul(z_rot, -point)
                
                # Changis axis
                transform = np.array([[0, 0, 1],
                                      [0, -1, 0],
                                      [1, 0, 0]])
                point = np.matmul(transform, point)

                point = point + camera_pos

                if point[2] < 0.03:
                    continue

                # Adding the point to the list of points
                points.append(point)

        points.append(camera_pos)

        return points



    def get_voxels(self, camera_frame, camera_pose):
        pass