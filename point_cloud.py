import numpy as np
import math
from math import cos, sin



class PointCloud:
    def __init__(self, shape, fovy, downsample=5) -> None:
        """
        Constructor for the PointCloud class.

        Parameters
        ----------
        shape : The shape of the depth frames from the camera.
        fovy : The field of view of the camera in the y-axis.
        """

        # Calculating intrinsics
        height, width = shape
        self.c_y = height / 2
        self.c_x = width / 2
        self.f_y = 0.5 * height / math.tan(fovy * math.pi / 360)
        self.f_x = self.f_y * (width / height)
        self.height = height
        self.width = width

        # Downsampling the depth frame for faster computation
        self.index_y = np.linspace(0, height-1, height//downsample, dtype=int)
        self.index_x = np.linspace(0, width-1, width//downsample, dtype=int)
    
    def get_feature_seg_vector(self, name, points):
        """
        Returns a feature vector for a given set of points.

        Parameters
        ----------
        name : Name of an object it detects
        points : A list of points in the world frame.
        
        """

        # Obtaining the average for each axis
        x = np.average(points[:,0])
        y = np.average(points[:,1])
        z = np.average(points[:,2])
        #print(len(points))


        # Creating feature vector
        feature_vector = np.array([x, y, z, 0, len(points)])
        
        # Adding the last value depending on the object
        if ("human" in name):
            feature_vector[3] = 0
        else:
            feature_vector[3] = np.random.randint(0, 1000)/10
        
        #print(feature_vector)

        return feature_vector
    
    def avg_feature_vector_seg (self, avg_feature_vectors):
        """
        Returns a weighted average feature vector for a given set of points.

        Parameters
        ----------
        avg_feature_vectors : A list of coordinates from the world frame.
        """

        # Feature vectors as a numpy array
        avg_feature_vectors = np.array(avg_feature_vectors)

        # Calculate the weight average of feature vectors
        x = np.average(avg_feature_vectors[:,0], weights=avg_feature_vectors[:,4])
        y = np.average(avg_feature_vectors[:,1], weights=avg_feature_vectors[:,4])
        z = np.average(avg_feature_vectors[:,2], weights=avg_feature_vectors[:,4])
        total_weight = np.sum(avg_feature_vectors[:,4])
        
        return np.array([x, y, z, 0, total_weight])

    
    def get_feature_vector(self, name, points):
        """
        Returns a feature vector for a given set of points.

        Parameters
        ----------
        name : Name of an object it detects
        points : A list of points in the world frame.
        """

        # Obtaining the average for each axis
        x = np.average(points[:,0])
        y = np.average(points[:,1])
        z = np.average(points[:,2])
        #print(len(points))


        # Creating feature vector
        feature_vector = np.array([x, y, z, 0])
        
        # Adding the last value depending on the object
        if ("human" in name):
            feature_vector[3] = 0
        else:
            feature_vector[3] = np.random.randint(0, 1000)/10
        
        #print(feature_vector)

        return feature_vector


    def get_feature_vectors(self, 
                            frames=None, 
                            segments=None, 
                            poses=None, 
                            rotations=None, 
                            m=None, 
                            segmented_map=None):
        """
        Returns a set of feature vectors for each agent in the map

        Parameters
        ----------
        frames : A list of depth frames from the camera.
        segments : A list of segmentation frames from the camera.
        poses : A list of camera poses in the world frame.
        rotations : A list of camera rotations in the world frame.
        m : The mujoco model.
        """

        if segmented_map is None:
            # Getting the segmented map
            segmented_map = self.get_segmented_map(frames, segments, poses, rotations, m)

        # Creating a list of feature vectors
        feature_vectors = {}

        # Iterating through each segmented object
        for key in segmented_map.keys():
                
            # Getting the feature vector
            feature_vector = self.get_feature_vector(key, np.array(segmented_map[key]))

            # Adding the feature vector to the list
            feature_vectors[key] = feature_vector

        return feature_vectors

    
    def get_segmented_map(self, frames, segments, poses, rotations, m):
        """
        Returns a set of feature vectors for each agent in the map

        Parameters
        ----------
        frames : A list of depth frames from the camera.
        segments : A list of segmentation frames from the camera.
        poses : A list of camera poses in the world frame.
        rotations : A list of camera rotations in the world frame.
        m : The mujoco model.
        """

        # Creating a list of points in the world frame
        segmented_point_cloud = {}

        # Iterating through each frame
        for i in range(len(frames)):

            # Getting the points in the world frame
            segment_points = self.get_segmented_points(frames[i], segments[i], poses[i], rotations[i])

            # Adding the list of points to the segmented point cloud list
            for key in segment_points.keys():

                 # Ignoring floor and world body
                if key == -1 or key == 0:
                    continue

                # Saving pure geoms
                if (m.geom(int(key)).bodyid[0] == 0):
                    try:
                        segmented_point_cloud[m.geom(int(key)).name].extend(segment_points[key])
                    except:
                        segmented_point_cloud[m.geom(int(key)).name] = segment_points[key]
                    continue

                # Saving geoms that are part of a body
                body_id = m.geom(int(key)).bodyid[0]
                while m.body(body_id).parentid != 0:
                    body_id = m.body(body_id).parentid[0]
                
                # Creating or extending the list of points for the body
                name = m.body(body_id).name
                try:
                    segmented_point_cloud[name].extend(segment_points[key])
                except:
                    segmented_point_cloud[name] = segment_points[key]


        return segmented_point_cloud


    def get_segmented_weightavg_feature_vector(self, frames, segments, poses, rotations, m):
        """
        Returns a set of feature vectors for each agent in the map

        Parameters
        ----------
        frames : A list of depth frames from the camera.
        segments : A list of segmentation frames from the camera.
        poses : A list of camera poses in the world frame.
        rotations : A list of camera rotations in the world frame.
        m : The mujoco model.
        """

        # Creating a list of points in the world frame
        segmented_point_cloud_vector = {}
        segmented_pc_feat_vec = {}
        segmented_point_cloud_percam = {}
        feature_vec_weigh_avg = {}


        # Iterating through each frame
        for i in range(len(frames)):

            # Getting the points in the world frame
            segmented_point_cloud_percam = self.get_segmented_points(frames[i], segments[i], poses[i], rotations[i])

            # Adding the list of points to the segmented point cloud list
            for key in segmented_point_cloud_percam.keys():

                 # Ignoring floor and world body
                if key == -1 or key == 0:
                    continue

                # Saving feature vectors of pure geoms from a camera
                if (m.geom(int(key)).bodyid[0] == 0):
                    try:
                        segmented_point_cloud_vector[m.geom(int(key)).name].append(self.get_feature_seg_vector(m.geom(int(key)).name, np.array(segmented_point_cloud_percam[key])))
                    except:
                        segmented_point_cloud_vector[m.geom(int(key)).name] = [self.get_feature_seg_vector(m.geom(int(key)).name, np.array(segmented_point_cloud_percam[key]))]

                    continue

                # Saving feature vectors of geoms that are part of a body
                body_id = m.geom(int(key)).bodyid[0]
                while m.body(body_id).parentid != 0:
                    body_id = m.body(body_id).parentid[0]
                
                # Listing feature vectors that belong to a single body
                name = m.body(body_id).name
                
                # Storing the feature vectors corresponding the body and part of it
                try:
                    segmented_point_cloud_vector[name].append(self.get_feature_seg_vector(name, np.array(segmented_point_cloud_percam[key])))

                except:
                    segmented_point_cloud_vector[name] = [self.get_feature_seg_vector(name, np.array(segmented_point_cloud_percam[key]))]


            # Getting weighted average of each feature vectors in a given frame
            for key in segmented_point_cloud_vector.keys():
                try:
                    segmented_pc_feat_vec[key].append(self.avg_feature_vector_seg(segmented_point_cloud_vector[key]))
                except:
                    segmented_pc_feat_vec[key] = [self.avg_feature_vector_seg(segmented_point_cloud_vector[key])]

        
        # Getting final weighted average of each feature vectors from all frames
        for key in segmented_pc_feat_vec.keys():
            feature_vec_weigh_avg[key] = self.avg_feature_vector_seg(segmented_pc_feat_vec[key])

        return feature_vec_weigh_avg


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
            

            # Adding the list of points to the point cloud list
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

        # Expanding depth frame for matrix multiplication
        depth_frame = np.expand_dims(depth_frame, axis=2)

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
        for i in self.index_y:
            for j in self.index_x:

                D = depth_frame[i][j]

                # Skip if the pixel is not in the image
                if np.isnan(D[0]):
                    continue

                # Obtaining world coordinates from depth pixels               
                mat = np.array([
                                [(i - self.c_y) / self.f_y],
                                [(j - self.c_x) * (self.width/self.height) / self.f_x],
                                [1]
                                ])
                point = np.matmul(mat, D)
                
                # Adjusting axis to match reconstruction
                transform = np.array([[0,  0, 1],
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
    

    def get_segmented_points(self, depth_frame, segment_frame, camera_pos, z_angle):
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

        # Expanding depth frame for matrix multiplication
        depth_frame = np.expand_dims(depth_frame, axis=2)

        # List of points in the world frame
        segment_points = {}

        # Creating rotation matrix
        z_angle = -z_angle - 180  # Adjusting for the camera's orientation
        rot = np.array([
                        [math.cos(z_angle * np.pi / 180), -math.sin(z_angle * np.pi / 180), 0],
                        [math.sin(z_angle * np.pi / 180), math.cos(z_angle * np.pi / 180), 0],
                        [0, 0, 1]
                        ])

        # Finding [x,y,z] coordinates of each pixel
        for i in self.index_y:
            for j in self.index_x:

                D = depth_frame[i][j]

                # Skip if the pixel is not in the image
                if np.isnan(D[0]) or segment_frame[i][j] <= 0:
                    continue

                # Obtaining world coordinates from depth pixels               
                mat = np.array([
                                [(i - self.c_y) / self.f_y],
                                [(j - self.c_x) * (self.width/self.height) / self.f_x],
                                [1]
                                ])
                point = np.matmul(mat, D)
                
                # Adjusting axis to match reconstruction
                transform = np.array([[0,  0, 1],
                                      [0, -1, 0],
                                      [-1, 0, 0]])
                point = np.matmul(transform, point)

                # Rotating the point to the world frame with pivot at the camera
                point = np.matmul(rot, point)

                # Translating the point to the world frame
                point = point + camera_pos

                if point[2] < 0.03:
                    continue

                # Adding points to their respective segemntation classes
                try:
                    segment_points[segment_frame[i][j]].append(point)
                except:
                    segment_points[segment_frame[i][j]] = [point]


        return segment_points


    def get_voxels(self, camera_frame, camera_pose):
        pass
