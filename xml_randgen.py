import time
import cv2
import mujoco as mj
import mujoco.viewer
import mujoco.glfw
import os 
import random
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
from sensors import Sensors

# notes: rolling friction may be too low, maybe should be velocity servo
# brownion motion and repulsion field for agent movement

def move_simple(name, *, t, func, axis=-1, model):
  model.geom(name).pos[axis] = func(t)


def main():
  import numpy as np

  iter = 0
  list_stl = []
  rootdir = "mjcf/dataset_ycb/models/"
  stl_file = None
  for subdir, dirs, files in os.walk(rootdir):
        if subdir.endswith("/google_16k") and iter < 120:
            new_subdir = subdir.replace("mjcf/dataset_ycb/models/", "")
            new_subdir = new_subdir.replace("/google_16k", "")
            for file in files:
                if file.endswith(".stl"):
                    stl_file = os.path.join(subdir, file)
                    list_stl.append(stl_file)
                    print(list_stl)
        if stl_file != None:
            iter += 1 
            
  for j in range(4):
      l = random.choice(list_stl)
      print(l)
      quat = np.random.random(4)
      quat = quat / np.linalg.norm(quat)
      rand = random.randint(0, 1)
      match rand:
        case 0:
          xml =f"""
              <mujoco>
                  <asset>
                      <mesh name="test" file="{l}"/>

                  </asset>
                  <worldbody>
                      <light name="top" pos="0 0 0.3"/>
                      <geom pos="0 -{quat[0]/5} 0.1" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="test"
                      />     
                      <camera name="camera" pos="0.5 0 0.1" xyaxes="0 0.5 0 -0.01 0 2" mode="fixed" fovy="60" />
                      <camera name="camera2" pos="0 0.5 0.1" xyaxes="-0.5 0 0 -0.01 0 2" mode="fixed" fovy="60" />
                      <camera name="camera3" pos="0 -0.5 0.1" xyaxes="0.5 0 0 -0.01 0 2" mode="fixed" fovy="60" />
                      <camera name="camera4" pos="-0.5 0 0.1" xyaxes=" 0 -0.5 0 -0.01 0 2" mode="fixed" fovy="60" />


                      <geom type="plane" size="1 1 0.1"/>   
                  </worldbody>
              </mujoco>    
              """
      
        case 1:
          l2 = random.choice(list_stl)
          xml =f"""
              <mujoco>
                  <asset>
                      <mesh name="test" file="{l}"/>
                      
                      <mesh name="test2" file="{l2}"/>

                  </asset>
                  <worldbody>
                      <light name="top" pos="0 0 0.3"/>
                      <geom pos="0 -{quat[0]/5} 0.1" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="test"
                      />   
                      <geom pos="0 0 0.1" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="test2"
                      />   
                      <camera name="camera" pos="0.5 0 0.1" xyaxes="0 0.5 0 -0.01 0 2" mode="fixed" fovy="60" />
                      <camera name="camera2" pos="0 0.5 0.1" xyaxes="-0.5 0 0 -0.01 0 2" mode="fixed" fovy="60" />
                      <camera name="camera3" pos="0 -0.5 0.1" xyaxes="0.5 0 0 -0.01 0 2" mode="fixed" fovy="60" />
                      <camera name="camera4" pos="-0.5 0 0.1" xyaxes=" 0 -0.5 0 -0.01 0 2" mode="fixed" fovy="60" />


                      <geom type="plane" size="1 1 0.1"/>   
                  </worldbody>
              </mujoco>    
              """
      
    
      
  
      m = mj.MjModel.from_xml_string(xml)
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
          #move_simple('human1', t=t, func=lambda x: (np.sin(x)+1)/2 * 0.2 + 0.05, axis=-1, model=m)     
          #print(m._camera_id2name[0])
          
          #print(mujoco.mj_id2name(m, 4, 0))

          mujoco.mj_step(m, d)
          viewer.sync()

          t += 0.1
          

          
          
          
      
          # Pick up changes to the physics state, apply perturbations, update options from GUI.
          ren_dep.enable_depth_rendering()
          ren_seg.enable_segmentation_rendering()
          #readings = sensor.get_rgbd_seg_matrices(m, d, ren_orig, ren_dep, ren_seg)
          #print(readings)
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
