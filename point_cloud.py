import numpy as np
import math



class PointCloud:
    def __init__(self, shape, fovy) -> None:
        """
        Constructor for the PointCloud class.

        Parameters
        ----------
        shape : The shape of the depth frames from the camera.
        fovy : The field of view of the camera in the y-axis.
        """
        self.shape = shape
        self.fovy = fovy


    def get_map(self, frames, poses, rotations):
        """
        Returns a numpy array of 3d points in the world frame from multiple 
        depth frames and cameras poses/rotations.

        Parameters
        ----------
        frames : A list of depth frames from the camera.
        poses : A list of camera poses in the world frame.
        rotations : A list of camera rotations in the world frame.
        """

        # Creating a list of points in the world frame
        point_cloud = []


        # Iterating through each frame
        for i in range(len(frames)):

            # Getting the points in the world frame
            points = self.get_points(frames[i], poses[i], rotations[i])

            # Adding the points to the point cloud
            point_cloud.extend(points)

        return np.array(point_cloud)



    def get_points(self, depth_frame, camera_pos, z_angle):
        """
        Returns a list of points in the world frame from a depth frame and camera pose.

        Parameters
        ----------
        depth_frame : The depth frame from the camera.
        camera_pos : The position of the camera in the world frame.
        rot : The rotation of the camera in the z-axis (extracted from 
        
        R.from_quat(m.body(m.cam_bodyid[i]).quat).as_euler('xyz', degrees=True)
        
        ).
        """

        # Camera intrinsics
        height, width = self.shape
        c_y = height / 2
        c_x = width / 2
        f_y = 0.5 * height / math.tan(self.fovy * math.pi / 360)
        f_x = f_y * (width / height)

        # Expanding depth frame for matrix multiplication
        depth_frame = np.expand_dims(depth_frame, axis=2)

        # Downsampling the depth frame for faster computation
        downsample = 2
        index_y = np.linspace(0, height-1, height//downsample, dtype=int)
        index_x = np.linspace(0, width-1, width//downsample, dtype=int)

        # List of points in the world frame
        points = []

        # Creating rotation matrix
        z_angle = -z_angle - 180  # Adjusting for the camera's orientation
        rot = np.array([
                        [math.cos(z_angle * np.pi / 180), -math.sin(z_angle * np.pi / 180), 0],
                        [math.sin(z_angle * np.pi / 180), math.cos(z_angle * np.pi / 180), 0],
                        [0, 0, 1]
                        ])

        # Finding [x,y,z] coordinates of each pixel
        for i in index_y:
            for j in index_x:

                D = depth_frame[i][j]

                # Skip if the pixel is not in the image
                if np.isnan(D[0]):
                    continue

                # Obtaining world coordinates from depth pixels               
                mat = np.array([
                                [(i - c_y) / f_y],
                                [(j - c_x) * (width/height) / f_x],
                                [1]
                                ])
                point = np.matmul(mat, D)
                
                # Adjusting axis to match reconstruction
                transform = np.array([[0, 0, 1],
                                      [0, -1, 0],
                                      [-1, 0, 0]])
                point = np.matmul(transform, point)

                # Rotating the point to the world frame with pivot at the camera
                point = np.matmul(rot, point)

                # Translating the point to the world frame
                point = point + camera_pos

                if point[2] < 0.03:
                    continue

                # Adding the point to the list of points
                points.append(point)

        return points



    def get_voxels(self, camera_frame, camera_pose):
        pass