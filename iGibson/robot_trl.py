"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import logging
import platform
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson.objects.ycb_object import YCBObject

from igibson.controllers import DifferentialDriveController # Left and right wheel vel 
from igibson.robots import REGISTERED_ROBOTS, ManipulationRobot # Robot registerd (turtle bot)
from igibson.scenes.empty_scene import EmptyScene # The scene for igibson and pybullet
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene # Need to download the dataset for this one
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene # Need to download the dataset for this one
from igibson.simulator import Simulator # Creates the simulation itself, either by pb and ig

CONTROL_MODES = OrderedDict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = OrderedDict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

GUIS = OrderedDict(
    ig="iGibson GUI (default)",
    pb="PyBullet GUI",
)

ARROWS = {
    0: "up_arrow",
    1: "down_arrow",
    2: "left_arrow",
    3: "right_arrow",
    65295: "left_arrow",
    65296: "right_arrow",
    65297: "up_arrow",
    65298: "down_arrow",
}

gui = "ig"


def choose_from_options(options, name, selection="user"):
    """
    Prints out options from a list, and returns the requested option.

    :param options: dict or Array, options to choose from. If dict, the value entries are assumed to be docstrings
        explaining the individual options
    :param name: str, name of the options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return str: Requested option
    """
    # Select robot
    print("\nHere is a list of available {}s:\n".format(name))

    for k, option in enumerate(options):
        docstring = ": {}".format(options[option]) if isinstance(options, dict) else ""
        print("[{}] {}{}".format(k + 1, option, docstring))
    print()

    if not selection != "user":
        try:
            s = input("Choose a {} (enter a number from 1 to {}): ".format(name, len(options)))
            # parse input into a number within range
            k = min(max(int(s), 1), len(options)) - 1
        except:
            k = 0
            print("Input is not valid. Use {} by default.".format(list(options)[k]))
    else:
        k = random.choice(range(len(options)))

    # Return requested option
    return list(options)[k]


def choose_controllers(robot, selection="user"):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return OrderedDict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = OrderedDict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = "DifferentialDriveController"#choose_from_options(options=options, name="{} controller".format(component), selection=selection)
        
        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


class KeyboardController:
    """
    Simple class for controlling iGibson robots using keyboard commands
    """

    def __init__(self, robot, simulator):
        """
        :param robot: BaseRobot, robot to control
        """
        # Store relevant info from robot
        self.simulator = simulator
        self.action_dim = robot.action_dim
        self.controller_info = OrderedDict()
        idx = 0
        for name, controller in robot._controllers.items():
            self.controller_info[name] = {
                "name": type(controller).__name__,
                "start_idx": idx,
                "command_dim": controller.command_dim,
            }
            idx += controller.command_dim

        # Other persistent variables we need to keep track of
        self.joint_control_idx = None  # Indices of joints being directly controlled via joint control
        self.current_joint = -1  # Active joint being controlled for joint control
        self.gripper_direction = 1.0  # Flips between -1 and 1
        self.persistent_gripper_action = None  # Whether gripper actions should persist between commands,
        # i.e.: if using binary gripper control and when no keypress is active, the gripper action should still the last executed gripper action
        self.last_keypress = None  # Last detected keypress
        self.keypress_mapping = None
        self.use_omnidirectional_base = robot.model_name in ["Tiago"]  # add other robots with omnidirectional bases
        self.populate_keypress_mapping()
        self.time_last_keyboard_input = time.time()

    def populate_keypress_mapping(self):
        """
        Populates the mapping @self.keypress_mapping, which maps keypresses to action info:

            keypress:
                idx: <int>
                val: <float>
        """
        self.keypress_mapping = {}
        self.joint_control_idx = set()

        # Add mapping for joint control directions (no index because these are inferred at runtime)
        self.keypress_mapping["]"] = {"idx": None, "val": 0.1}
        self.keypress_mapping["["] = {"idx": None, "val": -0.1}

        # Iterate over all controller info and populate mapping
        for component, info in self.controller_info.items():
            if self.use_omnidirectional_base:
                self.keypress_mapping["i"] = {"idx": 0, "val": 2.0}
                self.keypress_mapping["k"] = {"idx": 0, "val": -2.0}
                self.keypress_mapping["u"] = {"idx": 1, "val": 1.0}
                self.keypress_mapping["o"] = {"idx": 1, "val": -1.0}
                self.keypress_mapping["j"] = {"idx": 2, "val": 1.0}
                self.keypress_mapping["l"] = {"idx": 2, "val": -1.0}
            if info["name"] == "JointController":
                for i in range(info["command_dim"]):
                    ctrl_idx = info["start_idx"] + i
                    self.joint_control_idx.add(ctrl_idx)
            elif info["name"] == "DifferentialDriveController":
                self.keypress_mapping["i"] = {"idx": info["start_idx"] + 0, "val": 0.2}
                self.keypress_mapping["k"] = {"idx": info["start_idx"] + 0, "val": -0.2}
                self.keypress_mapping["l"] = {"idx": info["start_idx"] + 1, "val": 0.1}
                self.keypress_mapping["j"] = {"idx": info["start_idx"] + 1, "val": -0.1}
            elif info["name"] == "InverseKinematicsController":
                self.keypress_mapping["up_arrow"] = {"idx": info["start_idx"] + 0, "val": 0.5}
                self.keypress_mapping["down_arrow"] = {"idx": info["start_idx"] + 0, "val": -0.5}
                self.keypress_mapping["right_arrow"] = {"idx": info["start_idx"] + 1, "val": -0.5}
                self.keypress_mapping["left_arrow"] = {"idx": info["start_idx"] + 1, "val": 0.5}
                self.keypress_mapping["p"] = {"idx": info["start_idx"] + 2, "val": 0.5}
                self.keypress_mapping[";"] = {"idx": info["start_idx"] + 2, "val": -0.5}
                self.keypress_mapping["n"] = {"idx": info["start_idx"] + 3, "val": 0.5}
                self.keypress_mapping["b"] = {"idx": info["start_idx"] + 3, "val": -0.5}
                self.keypress_mapping["o"] = {"idx": info["start_idx"] + 4, "val": 0.5}
                self.keypress_mapping["u"] = {"idx": info["start_idx"] + 4, "val": -0.5}
                self.keypress_mapping["v"] = {"idx": info["start_idx"] + 5, "val": 0.5}
                self.keypress_mapping["c"] = {"idx": info["start_idx"] + 5, "val": -0.5}
            elif info["name"] == "MultiFingerGripperController":
                if info["command_dim"] > 1:
                    for i in range(info["command_dim"]):
                        ctrl_idx = info["start_idx"] + i
                        self.joint_control_idx.add(ctrl_idx)
                else:
                    self.keypress_mapping[" "] = {"idx": info["start_idx"], "val": 1.0}
                    self.persistent_gripper_action = 1.0
            elif info["name"] == "NullGripperController":
                # We won't send actions if using a null gripper controller
                self.keypress_mapping[" "] = {"idx": info["start_idx"], "val": None}
            else:
                raise ValueError("Unknown controller name received: {}".format(info["name"]))


def main(selection="user", headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """

    # Create an initial headless dummy scene so we can load the requested robot and extract useful info
    s = Simulator(mode="headless", use_pb_gui=False) # To see the simulation in 3D
    scene = EmptyScene()
    s.import_scene(scene)

    # Get robot to create
    robot_name = "Turtlebot" 
    robot = REGISTERED_ROBOTS[robot_name](action_type="continuous")
    
    # Import the robot to the simulation
    print(type(robot))
    s.import_object(robot)

    # For the second and further selections, we either as the user or randomize
    # If the we are exhaustively testing the first selection, we randomize the rest
    selection = "random"
    control_mode = "random"
    controller_choices = choose_controllers(robot=robot, selection=selection)
    print("controller_choices: ",controller_choices)

    # Choose scene to load
    scene_id = "empty" # You are choosing the empty scene


    # Choose GUI
    global gui
    gui = "ig"#"pb"#"ig" # Switch gui (pybullet or Igibson)

    if (
        gui == "ig"
        and platform.system() != "Darwin"
        and control_mode == "teleop"
        and isinstance(robot, ManipulationRobot)
        and "InverseKinematicsController" in controller_choices.values()
    ):
        logging.warning(
            "Warning: iG GUI does not support arrow keys for your OS (needed to control the arm with an IK Controller). Falling back to PyBullet (pb) GUI."
        )
        gui = "pb"

    # Infer what GUI(s) to use
    render_mode, use_pb_gui = None, None
    if gui == "ig":
        render_mode, use_pb_gui = "gui_interactive", False
    elif gui == "pb":
        render_mode, use_pb_gui = "headless", True
    else:
        raise ValueError("Unknown GUI: {}".format(gui))

    if headless:
        render_mode, use_pb_gui = "headless", False

    # Shut down dummy simulator and re-create actual simulator
    s.disconnect()
    del s

    ##################################################################### Start of simulation
    s = Simulator(mode=render_mode, use_pb_gui=use_pb_gui, image_width=512, image_height=512)

    # Load scene for simulation

    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1]) if scene_id == "empty" else InteractiveIndoorScene(scene_id)
    s.import_scene(scene)

    # Load robot
    robot = REGISTERED_ROBOTS[robot_name](
        action_type="continuous",
        action_normalize=True,
        controller_config={
            component: {"name": controller_name} for component, controller_name in controller_choices.items()
        },
    )
    
    s.import_object(robot)
    

    #################################################################### Reset the robot or spawn
    robot.set_position([0, 0, 0]) # -0.75, 1.0, 0]
    robot.set_orientation([0, 0 ,  0,  0.000001])#0.00532198, -0.0018555 ,  0.93055206,  0.36611624]
    robot.reset()
    robot.keep_still()


    # Set initial viewer if using IG GUI
    if gui != "pb" and not headless:
        s.viewer.initial_pos = [1.6, 0, 1.3]
        s.viewer.initial_view_direction = [-0.2, 0, -0.7]
        s.viewer.reset_viewer()
    
    

    # Create teleop controller
    action_generator = KeyboardController(robot=robot, simulator=s)

    # Other helpful user info
    print("Running demo. Switch to the viewer windows")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    # Desired positions
    desired_pos = np.array([2, 2, 0.01063783]) # coordinates x,y,z that goes to
    desired_degree = np.array([ 0.00528928,  0.00138672,  0.96720345, -0.25394405]) # quaternions
 
    
    velocities = np.array([])
    prev_error = 0
    prev_error_ang = 0
    angles = 0

    while step != max_steps:

    ################################################################################# Simulation
        
        angles = (Rotation.from_quat(robot.get_position_orientation()[1]).as_euler('zyx')[0])
        arc2 = np.arctan2(desired_pos[0], desired_pos[1])
        

        ############### Updated postions and errors
        x_pos = robot.get_position_orientation()[0][0] -  desired_pos[0]
        y_pos = robot.get_position_orientation()[0][1] -  desired_pos[1]

        # Correction of radians or angles
        if angles < 0:
            angles = angles + 2*np.pi
        elif angles > 6.28:
            angles = angles % 6.28
        if arc2 < 0:
            arc2 = arc2 + 2*np.pi
       
        error_ang = arc2 - angles


        #print("Postion and Orientations: ", robot.get_position_orientation())

        error = np.sqrt(np.power(x_pos,2) + np.power(y_pos,2))
        
        ################ Controller parameters
        # For angular
        kp=0.17 
        kd=0.05 

        # For linear
        kp_ =0.0035
        kd_ = 0.05

        lin_vel = error * kp_ + (error-prev_error)*kd_
        ang_vel = kp * error_ang + (error_ang - prev_error_ang)*kd
        
        command = [lin_vel,ang_vel]
        velocities = DifferentialDriveController._command_to_control(command, command, velocities) 
        lin_vel, ang_vel = velocities # Velocities for wheels


        if error_ang < 0.91 and error < 0.919: # Error threshold (its big but its just to make sure it stops)
            print("Reached final!")
            robot.apply_action([0, 0])
 
        else:
            robot.apply_action([lin_vel,-ang_vel])
       

        #robot.apply_action([0, ang_vel])
        

        prev_error = error
        prev_error_ang = error_ang
        
        for _ in range(10):
            s.step()
            step += 1




    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
