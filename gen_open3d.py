import open3d as o3d
import numpy as np

def __init__():
  pass

def drawpointcloud (self, xs, ys, ys, data):
        """
        Returns a visualization of the point clouds in open3d

        Parameters
        ----------
        xs : The x coordinate of the point cloud 
        ys : The y coordinate of the point cloud 
        zs : The z coordinate of the point cloud 
        data : Point cloud map
        """
        # Joins the coordinates
        xyz = np.zeros((np.size(data), 3))
        z = np.sinc((np.power(xs, 2) + np.power(ys, 2)))
        z_norm = (z - z.min()) / (z.max() - z.min())
        xyz = np.zeros((np.size(xs), 3))
        xyz[:, 0] = np.reshape(xs, -1)
        xyz[:, 1] = np.reshape(ys, -1)
        xyz[:, 2] = np.reshape(zs, -1)
        #print('xyz')
        #print(xyz)

        # Generates the point cloud from the coordinates 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # General transformation
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
        o3d.visualization.draw_geometries([pcd])
