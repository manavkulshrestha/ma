import time
import mujoco
import mujoco.viewer
import mujoco.renderer
import mujoco.glfw
import numpy as np
import open3d as o3d
import math
import matplotlib.image as mpimg


# Creating OpenGL context
ctx = mujoco.GLContext(1200, 900)
ctx.make_current()

# Creating xml mujoco object with a camera
# <camera name="camera" pos="0 0 0.12" zaxis="0 1 0" mode="fixed" fovy="60" />
#<joint name="free" type="free" axis="0 0 0" pos="0 0.2 0"/>
#            <geom type="sphere" size="0.1" rgba="0 0 1 1"/>

xml=""" 
<mujoco>
    <worldbody>
        <body name="floor" pos="0 0 0">

            <geom type="plane" size="1 1 0.1"/>

        </body>

        <body name="robot" pos="0 0 1">
            <joint name="free" type="free" axis="0 0 0" pos="0 0.2 0"/>
            <geom type="sphere" size="0.1" rgba="0 0 1 1"/>
            <camera name="camera" pos="0 0 1" zaxis="0 0 1" mode="fixed" fovy="60" />

        </body>

    </worldbody>

    <visual>

      <global offwidth="1920" offheight="1080"/>

    </visual>

</mujoco>
"""


# Creating a model from the xml and testing diferent cameras

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 900, 1200)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene()
con = mujoco.MjrContext()


steps = 0

# Launching the viewer
n_frames = 60
with mujoco.viewer.launch_passive(m, d) as viewer:

  # Setting the camera

  print(type(m.cam("camera")))
  print(type(viewer.cam))

  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  while viewer.is_running() and time.time() - start < 30:
  #for i in range(n_frames):

    step_start = time.time()

    
    #model = mujoco.MjModel.from_xml_string(xml)
    #sim = mp.MjSim(model)

    ## a is a tuple if depth is True and a numpy array if depth is False ##
    #a = sim.render(width=200, height=200, camera_name='camera', depth=True)
    #rgb_img = a[0]
    #depth_img = a[1]
    #print(rgb_img)

    # mj_step can be replaced with code that also evaluates


    # a policy and applies a control signal before stepping the physics.

    mujoco.mj_step(m, d)

    # update renderer to render depth
    r.enable_depth_rendering()
    # reset the scene
    r.update_scene(d, camera="camera")
    
    # depth is a float array, in meters.
    depth = r.render()
    # Scale by 2 mean distances of near rays.
    depth -= depth.min()
    depth /= 2*depth[depth <= 1].mean()
    # regla de 3 simple
    pixels = 255*np.clip(depth, 0, 1)
    # To use as an image
    pixels = pixels.astype(np.uint8)


    #r = r._depth_rendering

    # if steps < 200:
    #pixels = r.render()
    print(pixels)
    
    #plt.imshow(pixels)
    #plt.savefig('myimg.png', pixels)
    

    
    mpimg.imsave(f'myimg_1.png', pixels, cmap='gray')
    time.sleep(2) 

    # Example modification of a viewer option: toggle contact points every two seconds.

    with viewer.lock():

      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.

    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.

    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    #plt.imshow(pixels)
    
    if time_until_next_step > 0:

      time.sleep(time_until_next_step)

    time.sleep(.01)
    #plt.show()
# Freeing the context

r.disable_depth_rendering()
ctx.free()

