import numpy as np
import mujoco as mj
import mujoco.viewer
import time

# Utility class to run a simulation
# Runs a mujoco xml and allows to control it
class Sim:

    def __init__(self,
                 xml_stirng = None,
                 xml_path = None):
        
        if xml_stirng is not None:
            try:
                self.m = mj.MjModel.from_xml_string(xml_stirng)
            except Exception as e:
                print(e)
                return

        elif xml_path is not None:
            try:
                self.m = mj.MjModel.from_xml_path(xml_path)
            except Exception as e:
                print(e)
                return
        
        self.d = mj.MjData(self.m)

        # Setting up the initial value of the joints
        self.d.ctrl = [0, 1.21, -2.78,
                       0, 1.21, -2.78,
                       0, 1.21, -2.78,
                       0, 1.21, -2.78,]

        self.step = 0
        self.switch = True

    def start_debug(self):
        print(f"Control: {self.d.ctrl}")
        print(f"Type: {type(self.d.ctrl)}")
        print(f"Single leg: {self.d.ctrl[0:3]}")

    def run_debug(self):
        if self.step % 100 == 0:
            self.start_debug()

    # Abstract class to control the simulation
    def ctrl(self):
        if self.step == 200:
            single_leg_angles = self.d.ctrl[0:3]
            print(f"Single leg angles: {single_leg_angles}")

            self.d.ctrl = np.concatenate((single_leg_angles, 
                                          single_leg_angles, 
                                          single_leg_angles, 
                                          single_leg_angles),
                                          axis=None)
            self.step = 1

    def view_sim(self, debug=False):
        # Debugging
        if debug:
            self.start_debug()

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running():
                # Call the control function
                self.ctrl()

                # Debugging
                if debug:
                    self.run_debug()

                # Step the simulation
                self.step += 1
                mujoco.mj_step(self.m, self.d)
                viewer.sync()
                time.sleep(0.01)