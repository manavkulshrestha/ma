import time

import mujoco as mj
import mujoco.viewer

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

from utility import unit
from controller import Controller
from human import Human
from agent import Robot

# notes: rolling friction may be too low, maybe should be velocity servo
# brownion motion and repulsion field for agent movement

def sample_nonoverlaping(existing, low=-1, high=1):
  if not existing:
    return np.random.uniform(low, high, size=2)
  
  while True:
    ret = np.random.uniform(low, high, size=2)
    if (norm(np.array(existing) - ret, axis=1) > 0.1).all():
      return ret
    
def prepare_xml(path):
  with open(path, 'r') as f:
    mjcf = f.read()
      
    existing = []
    for i in range(5):
        xy = sample_nonoverlaping(existing)
        mjcf = mjcf.replace(f'<HPOS{i}>', f"{' '.join(map(str, xy))} 0.05")

    for i in range(5):
        xy = sample_nonoverlaping(existing)
        mjcf = mjcf.replace(f'<RPOS{i}>', f"{' '.join(map(str, xy))} 0.05")

  ret_filename = './mjcf/temp.xml'
  with open(ret_filename, 'w') as f:
    f.write(mjcf)

  return ret_filename

def main():
  mjcf_path = prepare_xml('./mjcf/simple.xml')
  m = mj.MjModel.from_xml_path(mjcf_path)
  d = mj.MjData(m)

  NUM_ROBOTS = 5
  NUM_HUMANS = 5

  # controller = Controller('robot', target=(-1, -1, 0), data=d, ksp=1, ksd=100)
  humans = [Human(f'human{i}', robots_num=NUM_ROBOTS, model=m, data=d) for i in range(NUM_HUMANS)]
  robots = [Robot(f'robot{i}', model=m, data=d) for i in range(NUM_ROBOTS)]

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
      time.sleep(0.001)


if __name__ == '__main__':
  main()