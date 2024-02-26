from utils import sim

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

simulator = sim.Sim(xml)

simulator.view_sim(debug=True)