import time

import mujoco
import mujoco.viewer
import mujoco.renderer
import mujoco.glfw

import matplotlib.pyplot as plt

# Debug flag
debug = False

# Creating OpenGL context
ctx = mujoco.GLContext(1200, 900)
ctx.make_current()

# Creating xml mujoco object with a camera
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

# Creating a model from the xml
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 900, 1200)

# Variable creation for MjvCamer (TODO: Learn how to use this)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene()
con = mujoco.MjrContext()

# Variables for pixel debugging
steps = 0
img_n = 0

# Launching the viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Setting the camera
  print(type(m.cam("camera")))
  print(type(viewer.cam))

  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()


  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)


    # Saving pixels to make sure sensor is working correctly
    r.update_scene(d, cam)
    if steps > 20 and debug:
      pixels = r.render()
      plt.imsave(f"./{img_n}.png", pixels)
      steps = 0
      img_n += 1
    steps += 1
      
    # pixels = r.render()
    # print(pixels)

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

# Freeing the context
ctx.free()