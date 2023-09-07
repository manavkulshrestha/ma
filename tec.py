import time

import mujoco as mj
import mujoco.viewer
import mujoco.renderer # Additional import

import numpy as np
from scipy.spatial.transform import Rotation as R

from utility import unit
from controller import Controller
from human import Human

from sensors import Sensors
from video_recorder import Recorder

# notes: rolling friction may be too low, maybe should be velocity servo
# brownion motion and repulsion field for agent movement


def move_simple(name, *, t, func, axis=-1, model):
  model.site(name).pos[axis] = func(t)


def main():
  m = mj.MjModel.from_xml_path('./mjcf/tec.xml')
  d = mj.MjData(m)
  t = 0

  # Create a renderer for data collection
  r_rgb = mujoco.renderer.Renderer(m, 480, 640)
  r_depth = mujoco.renderer.Renderer(m, 480, 640)
  r_seg = mujoco.renderer.Renderer(m, 480, 640)
  r_depth.enable_depth_rendering()
  r_seg.enable_segmentation_rendering()

  # Create a recorder for data collection
  sensor = Sensors()
  recorder = Recorder(duration=10, fps=30, n_cameras=m.cam_user.shape[0], folder="./")


  with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.

    start = time.time()
    while viewer.is_running() and time.time() - start < recorder.duration:
      step_start = time.time()

      # policy
      # mj_step can be replaced with code that also evaluates
      # a policy and applies a contro l signal before stepping the physics.

      move_simple('human1', t=t, func=lambda x: (np.sin(x)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      move_simple('human2', t=t, func=lambda x: (np.sin(x*2)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      move_simple('human3', t=t, func=lambda x: (np.sin(x*3)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      mujoco.mj_step(m, d)
      t += 0.1

      # Get RGB, depth, and segmentation images
      readings = sensor.get_rgbd_seg_matrices(m, d, r_rgb, r_depth, r_seg)
      # Save images to recorder
      recorder.add_frame(readings)
  
      # Pick up changes to the physics state, apply perturbations, update options from GUI.
      viewer.sync()

      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
      time.sleep(0.01)

    # Save videos
    recorder.save_videos()


if __name__ == '__main__':
  main()