import time

import mujoco
import mujoco.viewer
import mujoco.renderer
import mujoco.glfw
import matplotlib.pyplot as plt
import numpy as np
import sensors
import mediapy as media



# ------------ Debug flag ----------------
debug = False
test_name = "camera_depth"


# ------- Creating OpenGL context --------
ctx = mujoco.GLContext(1200, 900)
ctx.make_current()


# ------------ XML DEFINITION ------------
xml = """
<mujoco>
    <worldbody>
        <body name="floor" pos="0 0 0">
            <geom type="plane" size="1 1 0.1"/>
        </body>
        <body name="robot" pos="0 0 1">
            <joint type="free"/>
            <geom type="sphere" size="0.1" rgba="0 0 1 1"/>
            <camera name="camera" pos="0 0 1" zaxis="0 0 2" mode="fixed" fovy="60" />
        </body>
    </worldbody>
    <visual>
      <global offwidth="1920" offheight="1080"/>
    </visual>
</mujoco>
"""


# ------- MODEL AND DATA CREATION ---------
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r_rgb = mujoco.Renderer(m, 900, 1200)
r_depth = mujoco.Renderer(m, 900, 1200)
r_depth.enable_depth_rendering()
s = sensors.Sensors()


# ----- Variables for pixel debugging ------
steps = 0
img_n = 0

def save_image(name, pixels, renderer):
  plt.imsave(name, pixels)


# ----- Variables for video creation -------
n_cameras = m.cam_user.shape[0]
videos = [[] for _ in range(n_cameras)]


# Launching the viewer
with mujoco.viewer.launch_passive(m, d) as viewer:


  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()


  while viewer.is_running() and time.time() - start < 10:

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
    

    # ------ Debug Code for Video Saving ------
    pixels = s.get_rgbd_image_matrices(m, d, r_rgb, r_depth)
    for i in range(n_cameras):
      videos[i].append(pixels[i][:, :, :3])


    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)


    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()


    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)


    time.sleep(.01)

# Saving the videos
for i in range(n_cameras):
  media.write_video("video_cam_" + str(i) + ".mp4", videos[i], fps=30)


# Freeing the context
ctx.free()