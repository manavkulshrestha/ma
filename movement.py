import mujoco
import time
import mujoco_viewer
from mujoco import viewer
import numpy as np
from torch import inverse
from utility import unit
from controller import Controller


m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)
controller = Controller('FL_calf', target=(-1, -1, 0), data=d)
t = 0

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    with viewer.lock():

      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
    viewer.sync()
    d.ctrl = [ 0, 0, 0,   # Right top
                  0, 0, 0,   # Left top
                  0, 1, 0,   # Right bottom
                  0, 1, 0]  # Left bottom
    negativ = 0


    start = time.time() # ['FL_calf', 'FL_hip', 'FL_thigh', 'FR_calf', 'FR_hip', 'FR_thigh', 'RL_calf', 'RL_hip', 'RL_thigh', 'RR_calf', 'RR_hip', 'RR_thigh', 'trunk', 'world']"
    while viewer.is_running() and time.time() - start < 30:
      step_start = time.time()
      print(m.jnt_range)
      
      t += 0.005
      if t <= 0.7:
      # NOTE: calfs work as negatives
        d.ctrl = [ 0, 0, 0,   # Right top
                  0, 0, 0,   # Left top
                  0, 1, 0,   # Right bottom
                  0, 1, 0]  # Left bottom
      else :
                #Hip,  thigh,  calf
        d.ctrl = [ 0, -np.sin(t*np.pi)/2 +0.2, -np.sin(t*np.pi), 
                   0, np.sin(t*np.pi)/2 +0.2, 0, 
                   0, np.sin(t*np.pi) + 0.5, 0,
                   0, -np.sin(t*np.pi) + 0.5, -np.sin(t*np.pi) ]
    
      mujoco.mj_step(m, d)
      viewer.sync()
      

      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
      time.sleep(0.01)
