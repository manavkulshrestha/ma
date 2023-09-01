import time
import mujoco
import mujoco.viewer
import mujoco.renderer
import mujoco.glfw
import numpy as np
import open3d as o3d
import math
import matplotlib.image as mpimg

# Creating OpenGL context
ctx = mujoco.GLContext(1200, 900)
ctx.make_current()

# Xml mujoco object with a camera
xml=""" 
<mujoco>
    <worldbody>
        <body name="floor" pos="0 0 0">

            <geom type="plane" size="1 1 0.1"/>

        </body>

        <body name="robot" pos="0 0 1">
            <joint name="free" type="free" axis="0 0 0" pos="0 0.2 0"/>
            <geom type="sphere" size="0.1" rgba="0 0 1 1"/>
            <camera name="camera" pos="0 0 0.5" zaxis="0 0 1" mode="fixed" fovy="60" />

        </body>

    </worldbody>

    <visual>

      <global offwidth="1920" offheight="1080"/>

    </visual>

</mujoco>
"""

# Creating a model from the xml and testing diferent cameras

m = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(m)
r = mujoco.Renderer(m, 640, 480) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()


steps = 0
n_frames = 10
frame = 0
depth = 0
# Launching the viewer
with mujoco.viewer.launch_passive(m, data) as viewer:
  
  # Simulation time
  start_sim = data.time
  # Real time
  start = time.time()
  
  while viewer.is_running() and time.time() - start < 20:

    step_start = time.time()


    mujoco.mj_step(m, data)

    # a policy and applies a control signal before stepping the physics.
    # update renderer to render depth
    r.enable_depth_rendering()
      # reset the scene
    r.update_scene(data, camera="camera")
    
    # depth is a float array, in meters.
    pixels_dis = r.render()

    depth = r.render()
    # Scale by 2 mean distances of near rays.
    depth -= depth.min()
    depth /= 2*depth[depth <= 1].mean()
    # regla de 3 simple
    pixels = 255*np.clip(depth, 0, 1)
    # To use as an image
    pixels_depth = pixels.astype(np.uint8)

    print(pixels_dis)
    print(pixels_depth)
    

    if data.time - start_sim > 0.092: # makes 33 frames
    
      frame = frame +1
      start_sim = data.time

      #start = time.time()
      mpimg.imsave(f'myimg_{frame}.png', pixels_depth, cmap='gray')

      #time.sleep(2) 

    # Example modification of a viewer option: toggle contact points every two seconds.

    with viewer.lock():

      # enable contact visualization option:
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.

    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.

    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    
    if time_until_next_step > 0:

      time.sleep(time_until_next_step)

    time.sleep(.01)
# Freeing the context

r.disable_depth_rendering()
ctx.free()

