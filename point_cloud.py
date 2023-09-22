import numpy as np
import math



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
    
    def avg_feature_vector_seg (self, mat_avg):
        
        x = []
        y = []
        z = []
        b = []
        amount = []
        weighted_avg = []
        feature_vec_weigh_avg = {}
        
        for key in mat_avg.keys():
            print(mat_avg[key][0][0]) 
            for i in range(len(mat_avg[key])):

                            
                x.append(mat_avg[key][i][0])
                y.append(mat_avg[key][i][1])
                z.append(mat_avg[key][i][2])
                amount.append(mat_avg[key][i][4])
            print("x ",x)
                
        
            weig_avg_x = np.average(np.array(x,dtype=np.float128), weights=amount)
            weig_avg_y = np.average(np.array(y,dtype=np.float128), weights=amount)
            weig_avg_z = np.average(np.array(z,dtype=np.float128), weights=amount)
            feature_vec_weigh_avg[key] = [weig_avg_x, weig_avg_y, weig_avg_z, 0]
        
        return feature_vec_weigh_avg


    
    def get_feature_vector(self, name, points):
        """
        Returns a feature vector for a given set of points.

        Parameters
        ----------
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
        point_individual = {}
        point_individual_pc = {}
        nop = 0

        # Iterating through each frame
        for i in range(len(frames)):

            # Getting the points in the world frame
            segment_points = self.get_segmented_points(frames[i], segments[i], poses[i], rotations[i])
            #point_individual[i] = self.get_segmented_points(frames[i], segments[i], poses[i], rotations[i])

            # Adding the list of points to the segmented point cloud list
            for key in segment_points.keys():

                 # Ignoring floor and world body
                if key == -1 or key == 0:
                    continue

                # Saving pure geoms
                if (m.geom(int(key)).bodyid[0] == 0):
                    try:
                        segmented_point_cloud[m.geom(int(key)).name].extend(segment_points[key])
                        #point_individual_pc[[m.geom(int(key)).name]].extend(point_individual[i][key])
                    except:
                        segmented_point_cloud[m.geom(int(key)).name] = segment_points[key]
                        #point_individual_pc[m.geom(int(key)).name] = point_individual[i][key]
                    continue

                # Saving geoms that are part of a body
                body_id = m.geom(int(key)).bodyid[0]
                while m.body(body_id).parentid != 0:
                    body_id = m.body(body_id).parentid[0]
                
                # Creating or extending the list of points for the body
                name = m.body(body_id).name
                try:
                    segmented_point_cloud[name].extend(segment_points[key])
                    #point_individual_pc[name].extend(point_individual[i][key])
                except:
                    segmented_point_cloud[name] = segment_points[key]
                    #point_individual_pc[name] = point_individual[i][key]
            
            #for key in segment_points.keys():
            #    segment_points[key]
            #    nop += 1
            #print(nop)
                

        return segmented_point_cloud
    
    def get_feature_avgweig_vectors(
                            self, 
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
        feature_vectors_avg = {}

        # Iterating through each segmented object
        for key in segmented_map.keys():
            for i in range(len(segmented_map[key])):
                                    
                # Getting the feature vector
                    #feature_vector = self.get_feature_vector(key, np.array(segmented_map[name][key]))
                    j = 2
                    if segmented_map[key] != 'robot2':

                        print(segmented_map[j])
                    #feature_vectors_avg[key].append(feature_vector)

                # Adding the feature vector to the list
                #feature_vectors[key] = feature_vector

        return feature_vectors_avg




    def get_segmented_weigavg_map(self, frames, segments, poses, rotations, m):
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
        segmented_pc_f = {}
        segmented_point_cloudlast = {}
        point_individual = {}
        point_individual_pc = {}
        yes = {}
        no = 0


        # Iterating through each frame
        for i in range(len(frames)):

            # Getting the points in the world frame
            point_individual = self.get_segmented_points(frames[i], segments[i], poses[i], rotations[i])

            

            # Adding the list of points to the segmented point cloud list
            for key in point_individual.keys():

                 # Ignoring floor and world body
                if key == -1 or key == 0:
                    continue

                # Saving pure geoms
                if (m.geom(int(key)).bodyid[0] == 0):
                    try:
                        point_individual_pc[[m.geom(int(key)).name]].extend(point_individual[key])
                        segmented_point_cloud[m.geom(int(key)).name].append(self.get_feature_seg_vector(m.geom(int(key)).name, np.array(point_individual[key])))
                    except:
                        point_individual_pc[m.geom(int(key)).name] = point_individual[key]
                        segmented_point_cloud[m.geom(int(key)).name] = [self.get_feature_seg_vector(m.geom(int(key)).name, np.array(point_individual[key]))]
                    #segmented_point_cloudlast[m.geom(int(key)).name].append(segmented_point_cloud[m.geom(int(key)).name])

                    continue

                # Saving geoms that are part of a body
                body_id = m.geom(int(key)).bodyid[0]
                while m.body(body_id).parentid != 0:
                    body_id = m.body(body_id).parentid[0]
                
                # Creating or extending the list of points for the body
                name = m.body(body_id).name
                try:
                    point_individual_pc[name].extend(point_individual[key])
                    segmented_point_cloud[name].append(self.get_feature_seg_vector(name, np.array(point_individual[key])))

                except:
                    point_individual_pc[name] = point_individual[key]
                    segmented_point_cloud[name] = [self.get_feature_seg_vector(name, np.array(point_individual[key]))]
        
        
            #segmented_point_cloudlast[i] = self.avg_feature_vector_seg(segmented_point_cloud)
            # Equivalent

            for key in segmented_point_cloud.keys():
                try:
                    segmented_pc_f[key].append(segmented_point_cloud[key])
                except:
                    segmented_pc_f[key] = segmented_point_cloud[key]
                #print(segmented_point_cloud[key])

        #yes = self.avg_feature_vector_seg(segmented_pc_f)
        x = []
        y = []
        z = []
        amount = []
        i = 0

        for key in segmented_point_cloud.keys():
            print(len(segmented_point_cloud[key])) 
            for i in range(len(segmented_point_cloud[key])):
                print(segmented_pc_f[key][i][1])

                            
                x.append(segmented_point_cloud[key][i][0])
                y.append(segmented_point_cloud[key][i][1])
                z.append(segmented_point_cloud[key][i][2])
                amount.append(segmented_point_cloud[key][i][4])
            print("x ",x)
                
            
            weig_avg_x = np.average(np.array(x,dtype=np.float128), weights=amount)
            weig_avg_y = np.average(np.array(y,dtype=np.float128), weights=amount)
            weig_avg_z = np.average(np.array(z,dtype=np.float128), weights=amount)
            segmented_pc_f[key] = [weig_avg_x, weig_avg_y, weig_avg_z, 0]

        #segmented_point_cloudlast = segmented_pc_f
                
                #segmented_point_cloudlast[name].append(segmented_point_cloud[name])
            
                        #if segmented_point_cloud != segmented_point_cloudlast:

                #print(segmented_point_cloud[key][i]) # x

                #print(point_individual_pc[key])
        

            
                #print(segmented_point_cloud)
                #print("")
                #no += 1
        #print("num total ")
        #print( no)
        return segmented_pc_f
                    

                
            
                

               
                                    

        
                

         
                
        

        
    



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