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
    

    # ------ Debug Code for Video Saving ------
    n_frames += 1
    pixels = s.get_rgbd_seg_matrices(m, d, r_rgb, r_depth, r_seg)
    print(pixels[0].shape)
    print(pixels[0])
    for i in range(n_cameras):
      videos[i].append(pixels[i][:, :, :3])


    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()


    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = abs(m.opt.timestep - (time.time() - step_start))

    current_time = time.time() - start
    while d.time < current_time:
      mujoco.mj_step(m, d)

    if time_until_next_step > 0:
      time.sleep(time_until_next_step)


# Saving the videos
print("Saving videos...")
for i in range(n_cameras):
  media.write_video("video_cam_" + str(i) + ".mp4", videos[i], fps=fps)