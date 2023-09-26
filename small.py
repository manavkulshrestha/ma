import time

import mujoco as mj
import mujoco.viewer

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

from utility import unit

from pathlib import Path
from data import prepare_episode, TEMPLATES_DIR
from agent import Human, Robot

EPISODES_PATH = Path('./mjcf/episodes/')
np.random.seed(42)

def main():
  mjcf_path = EPISODES_PATH/'small.xml'
  mjcf, num_robots, num_humans = prepare_episode(TEMPLATES_DIR/'env_template.xml',
                                                 r_nrange=(3, 6), h_nrange=(3, 6), env_bounds=(-1, 1),
                                                 simple_robots=True,
                                                 episode_path=mjcf_path)
  m = mj.MjModel.from_xml_path(str(mjcf_path))
  d = mj.MjData(m)

  humans = [Human(f'human{i}', robots_num=num_robots, model=m, data=d) for i in range(num_humans)]
  robots = [Robot(f'robot{i}', model=m, data=d) for i in range(num_robots)]

  with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
      step_start = time.time()

      # policy
      for h in humans:
        h.wander()
      for r in robots:
        r.herd()

      mujoco.mj_step(m, d)
      viewer.sync()

      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
      # time.sleep(0.001)


if __name__ == '__main__':
  main()