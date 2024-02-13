import time
import cv2
import mujoco as mj
import mujoco.viewer
import mujoco.glfw

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from utility import unit
from controller import Controller

from human import Human
import time
import mujoco

import mujoco.renderer
import mujoco.glfw
import numpy as np
import open3d as o3d
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from scripts.sensors import Sensors

# notes: rolling friction may be too low, maybe should be velocity servo
# brownion motion and repulsion field for agent movement

def move_simple(name, *, t, func, axis=-1, model):
  model.site(name).pos[axis] = func(t)

def ConvexHull_points(mat):
  print("ConvexHull")
  points_hull = mat[:,0:3]

  hull = ConvexHull(points_hull)
  centroid = np.mean(points_hull[hull.vertices, :], axis=0)

  x = np.append(mat[:, 0], centroid[0])
  y = np.append(mat[:, 1], centroid[1])
  z = np.append(mat[:, 2], centroid[2])
  print(centroid[0])
  print(np.mean(x))
  print(centroid[1])
  print(np.mean(y))
  print(centroid[2])
  print(np.mean(z))
  
  return x, y, z




def main():
  

  m = mj.MjModel.from_xml_path('./mjcf/tec.xml')
  d = mj.MjData(m)
  r = mujoco.Renderer(m, 480, 640) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480
  ren_orig = mujoco.Renderer(m, 480, 640) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480
  ren_dep = mujoco.Renderer(m, 480, 640) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480
  ren_seg = mujoco.Renderer(m, 480, 640) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480
  opt = mujoco.MjvOption()
  opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
  sensor = Sensors()


  frame = 0

  cam = mujoco.MjvCamera()
  result = {}
  poses = []
  rotations = []
  points = []

  point = []
  point_cloud = []

  t = 0
  start_sim = d.time
  no = -1
  z_angle = 0

  with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    with viewer.lock():

      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
    viewer.sync()
    


    start = time.time()
    while viewer.is_running() and time.time() - start < 2:
      step_start = time.time()
    
    
      # policy
      # mj_step can be replaced with code that also evaluates
      # a policy and applies a contro l signal before stepping the physics.

      #move_simple('human1', t=t, func=lambda x: (np.sin(x)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      #move_simple('human2', t=t, func=lambda x: (np.sin(x*2)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      #move_simple('human3', t=t, func=lambda x: (np.sin(x*3)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      mujoco.mj_step(m, d)
      viewer.sync()

      t += 0.1
      

      
      
      
  
      # Pick up changes to the physics state, apply perturbations, update options from GUI.
      ren_dep.enable_depth_rendering()
      ren_seg.enable_segmentation_rendering()
      readings = sensor.get_rgbd_seg_matrices(m, d, ren_orig, ren_dep, ren_seg)
      print(readings)
      #print(range(m.cam_user.shape[0]))

      

      #geom_ids = seg[:, :, 0]
      #print(geom_ids)

      #print(geom_ids.shape)
      
      
      #depth_r = r.render()

      #print(depth_r2)
      #depth = np.expand_dims(depth_r, axis=2)
      #depth = np.expand_dims(depth_r, axis=2)
      #depth[depth < 0.01] = np.nan
      #depth[depth > 4.0] = np.nan
      #print(depth.shape)
      #print(depth_r.shape)
      #print(depth)
      
      #print(result[no].shape)

      if d.time - start_sim > 0.092:
    
        frame = frame +1
        start_sim = d.time

      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
      time.sleep(0.01)
    ren_dep.disable_depth_rendering()
    ren_seg.disable_segmentation_rendering()
    
    frames = [readings[i][:, :, 3] for i in range(len(readings))]
    segments = [readings[i][:, :, 4] for i in range(len(readings))]

    poses = []
    rotations = []
    for i in range(m.cam_user.shape[0]):
    # Get the position of the camera
      pos = m.cam_pos[i] + m.body(m.cam_bodyid[i]).pos
      poses.append(pos)

      # Get the rotation of the camera
      angle = R.from_quat(m.body(m.cam_bodyid[i]).quat).as_euler('xyz', degrees=True)
      rotations.append(angle[0])

    import point_cloud as pcu

    point_cloud = pcu.PointCloud((480, 640), 60, downsample=2)
    mat = point_cloud.get_map(frames, poses, rotations)
    

    segemented_mat = point_cloud.get_segmented_map(frames, segments, poses, rotations, m)
    mat_avg = point_cloud.get_segmented_weigavg_map(frames, segments, poses, rotations, m)
    #print(mat_avg["human3"][0][0])


    #for i in range(3):
      #for j in range(len(mat_avg)):
        #feature_vector = point_cloud.get_feature_vector(key, np.array(mat_avg[i][j]))

    #print(len(mat_avg))

    feature_vectors = point_cloud.get_feature_vectors(segmented_map=segemented_mat)
    #print(feature_vectors_avg)
    #feat_vec_mat = point_cloud.get_feature_avgweig_vectors(m = m,segmented_map=mat_avg)
    
    x, y, z = ConvexHull_points(mat)
    
    size = 0
    size2 = 0
    #for key in segemented_mat:
    #    size += len(mat_avg[key])
    #    size2 += len(segemented_mat[key])
        



    print("Total -> ", mat.shape)
    print("Segmented -> ", size)
    print("Segmented original-> ", size2)
    
    """


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Define the x, y, z coordinates of the point cloud
    x = mat[:, 0]
    y = mat[:, 1]
    z = mat[:, 2]

    # Plot the point cloud data
    ax.scatter(x, y, z, s=1)

    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    for key in segemented_mat.keys():
        mat = np.array(segemented_mat[key])
        print(mat.shape)

        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.suptitle(key)

        # Define the x, y, z coordinates of the point cloud
        x = mat[:, 0]
        y = mat[:, 1]
        z = mat[:, 2]

        # Plot the point cloud data
        ax.scatter(x, y, z, s=1)

        # Set the axis labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Show the plot
        plt.show()

    """

    """

    
    result[0] = np.append(original, depth, axis=2)
    result[0] = np.append(result[0], r_seg, axis=2)

    result[1] = np.append(original2, depth_r1, axis=2)
    result[1] = np.append(result[1], r1_seg, axis=2)
    


  for k in range (1):
    pos = m.cam_pos[k] + m.body(m.cam_bodyid[k]).pos
    poses.append(pos)

    #angle = R.from_quat(m.body(m.cam_bodyid[1]).quat).as_euler('xyz', degrees=True)
    #rotations.append(angle[0])
    #print(result[no])

    # To use as an image
    angle = R.from_quat(m.body(m.cam_bodyid[k]).quat).as_euler('xyz', degrees=True)
    #rotations.append(angle[0])
    print(angle[0])


    # Expanding depth frame for matrix multiplication
    depth_frame = np.array(result[k][:, :, 3])
    depth_frame = np.expand_dims(depth_frame, axis=2)
    print(depth_frame.shape)

    seg_frame = np.array(result[k][:, :, 4])
    seg_frame = np.expand_dims(seg_frame, axis=2)
    seg_frame = np.array(seg_frame.reshape(480,640))
    #print(seg_frame.reshape(480,640).shape)
    


    
    height = 480
    width = 640
    c_y = height / 2
    c_x = width / 2
    f_y = 0.5 * height / math.tan(60 * math.pi / 360)
    f_x = f_y * (width / height) 

    segment_points = {}


      # Downsampling the depth frame for faster computation
    downsample = 5
    index_y = np.linspace(0, height-1, height//downsample, dtype=int)
    index_x = np.linspace(0, width-1, width//downsample, dtype=int)

      # List of points in the world frame

      # Creating rotation matrix
    z_angle = angle[0]
    z_angle = -z_angle - 180  # Adjusting for the camera's orientation
    rot = np.array([
                      [math.cos(z_angle * np.pi / 180), -math.sin(z_angle * np.pi / 180), 0],
                      [math.sin(z_angle * np.pi / 180), math.cos(z_angle * np.pi / 180), 0],
                      [0, 0, 1]
                      ])

    for i in index_y:

              for j in index_x:

                  D = depth_frame[i][j]

                    # Skip if the pixel is not in the image
                  if np.isnan(D[0]) or seg_frame[i][j] <= 0:
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
                  point = point + pos
                  #point = np.reshape(point[0],(1,3))
                  #print(point.shape)



                  if point[2] < 0.03:
                      continue
                  #print(point.shape)

                  # Adding the point to the list of points
                  points.append(point)

                  #print(seg_frame.shape)
                  

                  #seg_frame[i][j] = np.append(point)

    mat = np.array(points)

      #print(mat.shape)


            #point_cloud.extend(points)
            #print(point.shape)
            #mat = np.array(point_cloud)
            #print(len(points[1]))


            #mat = np.delete(mat,0,1)
            #newmat = mat[[0, 2], :]


            #print(newmat.shape)


#mat = np.array(segemented_mat[key])
#print(mat.shape)


            
            
      
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  x = mat[:, 0]
  y = mat[:, 1]
  z = mat[:, 2]



        
        
        # Define the x, y, z coordinates of the point cloud
  
  # Calculate geometric centroid of convex hull 

      # Plot the point cloud data
      # Pick a range of columns new[:,0:3]
      

  ax.scatter(x, y, z, s=1)

      # Set the axis labels
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

      # Show the plot
  plt.show()
  """
  



if __name__ == '__main__':
  main()
