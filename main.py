import time

import mujoco as mj
import mujoco.viewer

import numpy as np
from scipy.spatial.transform import Rotation as R

from utility import unit
from controller import Controller
# from human import Human

# notes: rolling friction may be too low, maybe should be velocity servo
# brownion motion and repulsion field for agent movement

def main():
  m = mj.MjModel.from_xml_path('./mjcf/main.xml')
  d = mj.MjData(m)

  # controller = Controller('robot', target=(-1, -1, 0), data=d, ksp=1, ksd=100)
  # human = Human('human', robots_num=1, model=m, data=d)

  with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
      step_start = time.time()

      # policy
      d.ctrl = [0.01, 0.01]
    #   human.wander()

      mujoco.mj_step(m, d)
      # m.geom('geom_object').pos *= 2; mujoco.mj_step(m, d); viewer.sync()
  
      viewer.sync()

      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
      time.sleep(0.001)


if __name__ == '__main__':
  main()