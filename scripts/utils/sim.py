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
        self.d.ctrl = [0, 0, -0.888,
                       0, 0, -0.888,
                       0, 0, -0.888,
                       0, 0, -0.888,]

        self.step = 0
        self.switch = True

    def start_debug(self):
        print(f"Actuator: {self.d.act}")
        print(f"Control: {self.d.ctrl}")

    def run_debug(self):
        if self.step % 100 == 0:
            self.start_debug()
            self.switch = not self.switch
            self.step += 1
        else:
            self.step += 1

        self.ctrl()

    # Abstract class to control the simulation
    def ctrl(self):
        pass
        # if self.switch:
        #     self.d.ctrl = [0, 0, -0.888,
        #                    0, 0, -0.888,
        #                    0, 0, -0.888,
        #                    0, 0, -0.888,]
        # else:
        #     self.d.ctrl = [0, 0, -2,
        #                    0, 0, -2,
        #                    0, 0, -2,
        #                    0, 0, -2,]

    def view_sim(self, debug=False):
        # Debugging
        if debug:
            self.start_debug()

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running():
                # Call the control function
                # self.ctrl()

                # Debugging
                if debug:
                    self.run_debug()

                # Step the simulation
                mujoco.mj_step(self.m, self.d)
                viewer.sync()
                time.sleep(0.01)