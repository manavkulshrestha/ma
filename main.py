import time

import mujoco as mj
import mujoco.viewer

import numpy as np
from scipy.spatial.transform import Rotation as R

from utility import unit
from controller import Controller
from human import Human

# notes: rolling friction may be too low, maybe should be velocity servo


def main():
  m = mj.MjModel.from_xml_path('./mjcf/main.xml')
  d = mj.MjData(m)

  controller = Controller('robot', target=(-1, -1, 0), data=d, ksp=1, ksd=100)
  human = Human('human', model=m)

  with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.

    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
      step_start = time.time()

      # policy
      # d.ctrl = [0.01, -0.01]
      # print(controller.get_orn2D())
      # d.ctrl = controller.get_control(scale=0.01)
      # print(f' {d.ctrl}')
      # print(f'{d.ctrl} at {np.around(controller.get_pos(), 3)}')
      # print(f'{controller.e_curr:.3f} {d.ctrl}')

      # mj_step can be replaced with code that also evaluates
      # a policy and applies a contro l signal before stepping the physics.
      human.wander()
      mujoco.mj_step(m, d)
  
      # Example modification of a viewer option: toggle contact points every two seconds.
      # with viewer.lock():
      #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

      # Pick up changes to the physics state, apply perturbations, update options from GUI.
      viewer.sync()

      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
      time.sleep(0.01)


if __name__ == '__main__':
  main()