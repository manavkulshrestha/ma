import mujoco as mj
import mujoco.viewer
import time

xml = """
<mujoco>
    <worldbody>
        <body name="floor" pos="0 0 0">
            <geom type="plane" size="1 1 0.1"/>
        </body>
            <body name="box" pos="0 0 0.5">
                <freejoint/>
                <geom type="box" size="0.2 0.2 0.2"/>
            </body>
    </worldbody>
</mujoco>
"""

m = mj.MjModel.from_xml_string(xml)
d = mj.MjData(m)

def move_simple(name, *, t, func, axis=-1, model):
  model.site(name).pos[axis] = func(t)

mujoco.mj_kinematics(m, d)
print('raw access:\n', d.geom_xpos)
print('\nnamed access:\n', d.geom(0).xpos)

print("\nxpos:\n", d.geom_xpos, "\n",
      "\nqpos:\n", d.joint(0), "\n")

i = 0

with mujoco.viewer.launch_passive(m, d) as viewer:

    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        i += 1

        if i == 500:
            d.body(2).xpos = [0, 0, -0.5]
            d.joint(0).qpos = [0, 0, 2, 1, 0, 0, 0]
        elif i == 501:
            print(d.body(2).xpos)
            i = 0

        mujoco.mj_step(m,d)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        time.sleep(0.01)