import time
import mujoco
import mujoco.viewer
import mujoco.renderer
import mujoco.glfw
import matplotlib.pyplot as plt
import numpy as np
import sensors
import mediapy as media

import os



# ------------ Debug flag ----------------
debug = False
test_name = "camera_depth"


# ------------ XML DEFINITION ------------
xml = """
<mujoco>
     <asset>
        <mesh name="test" file="ycb/063-a_marbles.stl" scale="1 1 1"/>
     </asset>

    <worldbody>
        <light name="top" pos="0 0 1"/>

        <geom pos="0 0 0" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="test"/>        
    </worldbody>
</mujoco>
"""


# ------- MODEL AND DATA CREATION ---------
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r_rgb = mujoco.Renderer(m, 480, 640)

r_depth = mujoco.Renderer(m, 480, 640)
r_depth.enable_depth_rendering()

r_seg = mujoco.Renderer(m, 480, 640)
r_seg.enable_segmentation_rendering()

s = sensors.Sensors()


# ----- Variables for pixel debugging ------
steps = 0
img_n = 0

def save_image(name, pixels, renderer):
  plt.imsave(name, pixels)


# ----- Variables for video creation -------
n_cameras = m.cam_user.shape[0]
videos = [[] for _ in range(n_cameras)]
duration = 5
fps = 30
n_frames = 1
# Set the model timestep to match the video framerate.
m.opt.timestep = 1 / fps


# Launching the viewer
with mujoco.viewer.launch_passive(m, d) as viewer:


  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()


  while viewer.is_running() and n_frames < duration * fps:

    step_start = time.time()


    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)


    # ------ Debug Code for Image Saving ------
    if debug:
      # Saving pixels to make sure sensor is working correctly
      r_rgb.update_scene(d)
      r_depth.update_scene(d)

      # Saving pixels to make sure sensor is working correctly
      if steps > 10:
        save_image("rgb_" + str(img_n) + ".png", r_rgb.render())
        steps = 0
        img_n += 1
      steps += 1

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()


    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = abs(m.opt.timestep - (time.time() - step_start))

    current_time = time.time() - start
    while d.time < current_time:
      mujoco.mj_step(m, d)

    if time_until_next_step > 0:
      time.sleep(time_until_next_step)