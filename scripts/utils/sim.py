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

    def start_debug(self):
        pass

    def run_debug(self):
        pass

    # Abstract class to control the simulation
    def ctrl(self):
        pass

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
                mujoco.mj_step(self.m, self.d)
                viewer.sync()
                time.sleep(0.01)