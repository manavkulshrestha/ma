import numpy as np
from math import tan, pi



class VectorizedPC:

    def __init__(self, shape, fovy):
        w, h = shape

       # Define intrinsic values of camera
        h, w = shape
        fovy = 60
        f_y = 0.5 * h / tan(fovy * pi / 360)
        f_x = f_y * (w / h)

        # Create arrays for x and y coordinates
        x = (np.arange(0, w) - w/2) * w / h / f_x
        y = (np.arange(0, h) - h/2) / f_y
        z = np.ones((w*h, 1))
        
        # Use meshgrid to create two grids of x and y coordinates
        X, Y = np.meshgrid(x, y)

        # Stack X and Y to create the matrix
        matrix = np.vstack((X.flatten(), Y.flatten())).T

        # Vectorized matrix to convert depth to 3D points
        self.matrix = np.concatenate((matrix, z), axis=1)

        # Matrix for tranforming axis to world frame
        self.transform = np.array([
            [0, 0, -1], 
            [-1, 0, 0], 
            [0, 1, 0]])



    def get_points(self, 
                   frame:np.ndarray, 
                   rotation:np.ndarray, 
                   position:np.ndarray,
                   downsample:int=1) -> np.ndarray:
        """
        Returns a list of points representing the point cloud 
        generated from the frame.
        
        Parameters
        ----------
        frame : numpy.ndarray
            The frame to get points from 
        rotation : numpy.ndarray
            The rotation matrix of the frame
        position : numpy.ndarray
            The position vector of the frame
        """
        
        # Flatten the frame
        frame = frame.reshape(-1, 1)

        # Get the points in the frame
        points = frame * self.matrix

        # Remove points with nan
        points = points[~np.isnan(points).any(axis=1)]

        # # Rotate the points
        points = points @ rotation

        # # Transform the points to the world frame
        points = points @ self.transform

        # # Translate the points
        points = points + position

        # Downsample the points
        # points = points[::downsample]

        return points
    


    def get_segmented_points(self,
                             frame:np.ndarray,
                             segmentation:np.ndarray,
                             rotation:np.ndarray,
                             position:np.ndarray,
                             downsample:int=1) -> np.ndarray:
        """
        Returns a list of points representing the point cloud
        generated from the frame.
        
        Parameters
        ----------
        frame : numpy.ndarray
            The frame to get points from
        segmentation : numpy.ndarray
            The segmentation array
        rotation : numpy.ndarray
            The rotation matrix of the frame
        position : numpy.ndarray
            The position vector of the frame
        """

        #  Gettingg all posible values of the segmentation array
        values = np.unique(segmentation)

        segmented_pc = {}

        frame[segmentation == -1] = np.nan
        for value in values:
            frame_copy = frame.copy()
            frame_copy[segmentation != value] = np.nan
            segmented_pc[value] = self.get_points(frame_copy, 
                                                  rotation, 
                                                  position, 
                                                  downsample)

        return segmented_pc
    


    def centroids(self, pc):
        """
        Returns the centroids of the objects from the point cloud
        generated from the frame.
        
        Parameters
        ----------
        pc : numpy.ndarray
            The point cloud for each object
        """
        centroids = []
        # Calculating the mean in x, y ,z for the point cloud of the object
        for i in list(pc.keys()):
            centroids.append(np.mean(pc[i], axis=0))
        
        return centroids

    
    
    def move_simple(self, *, t, func, axis=-1, model):
        """
        Moves the objects of the simulation in relation of the time       
        Parameters
        ----------
        t : float
            Time for the iteration
        func : function
            Defines the movement of the object
        axis : int
            Defines the direction of the movement
        model : MjModel
            Model of the scene
        """
        # Iterates over the geoms and moves them
        for i in range(model.ngeom):
            model.geom(i).pos[axis] += func(t)

        return model



    def calc_vel(self, fin, ini, times, m):
        """
        Moves the objects of the simulation in relation of the time       
        Parameters
        ----------
        times : list
            Has the inicial and final time of movement
        fin : list
            Centroids of the geoms in the last frame
        ini : int
            Centroids of the geoms in the first frame
        model : MjModel
            Model of the scene
        """
        calc_v = {}
        # Iterates through the geom's initial position and final position 
        for i in range(m.ngeom):
            # Calculates velocity
            calc_vel = (fin[i] - ini[i])/(times[1]-times[0])
            
            # Puts together in a dictionary the postion and velocity per object
            calc_v[m.geom(i).name] = np.append([ini[i]], calc_vel[0:3])


        return calc_v


        
        
   

        