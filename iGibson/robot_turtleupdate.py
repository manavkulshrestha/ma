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
import keyboard

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson.controllers import DifferentialDriveController # Left and right wheel vel 
from igibson.robots import REGISTERED_ROBOTS, ManipulationRobot # Robot registerd (turtle bot)
from igibson.scenes.empty_scene import EmptyScene # The scene for igibson and pybullet
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene # Need to download the dataset for this one
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene # Need to download the dataset for this one
from igibson.simulator import Simulator # Creates the simulation itself, either by pb and ig


from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

def create_trajectory(curr_pos, bounds=[0,1], num_keypoints=10, length=1000, display=False):
    '''
    Generates a smooth trajectory of given size, starting at the given location, within the given bounds.

    curr_pos = [x=float, y=float]. location for where to start the trajectory. this should be your robot's initial position
    bounds = [lower=float, upper=float]. bounds for your environment. the trajectory generated will try to be inside this
    num_keypoints = int. The number of random locations to visit and incorporate in the trajectory
    length = int. The desired length for the returned trajectory.
    display = True/False. plots the generated trajectory and generated keypoints

    returns: spline = [[x0=float,y0=float],...,[xn=float,yn=float]] where n = length (argument given).
    This sequence of points is a smooth trajectory passing through all the randomly generated keypoints.
    '''
    curr_pos = [curr_pos[0], curr_pos[1]]

    keypoints = np.random.uniform(*bounds, size=[2, num_keypoints])
    keypoints[:, 0] = curr_pos

    tck, _ = splprep(keypoints, k=2, s=0) # k=2 makes a c2 spline, s=0 looks good

    t = np.linspace(0, 1, length)
    spline = splev(t, tck)

    if display:
        plt.plot(*keypoints, 'o', label='Original Points')
        plt.plot(*spline, label='Cubic Spline')
        plt.show()

    return spline, keypoints



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
        self.time_last_keyboard_input = time.time()

   
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
    gui = "pb"#"pb" # Switch gui (pybullet or Igibson)

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
    
    
    # Other helpful user info
    print("Running demo. Switch to the viewer windows")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    # Desired positions
    velocities = np.array([])
    targets = np.array([])
    spline = np.array([])

    prev_error = 0
    prev_error_ang = 0
    angles = 0
    
    vs_id_spline = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 1, 1, 1])
    vs_id_keypoints = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 1, 0, 1])
    vs_id_target = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])



    trajectory, targets = create_trajectory(curr_pos=robot.get_position_orientation()[0], bounds=[-1,1], display=True)
    trajectory = np.multiply(trajectory, 10)
    targets = np.multiply(targets, 10)

    # Plotting the trajectory
    for i in range(0, trajectory[0].size, 20):
        x = trajectory[0][i] 
        y = trajectory[1][i]
        p.createMultiBody(basePosition=[trajectory[0][i], trajectory[1][i], 0.01063783], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id_spline)

    # Plotting the keypoints
    for i in range(0, targets[0].size):
        p.createMultiBody(basePosition=[targets[0][i], targets[1][i], 0.01063783], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id_keypoints)
    
    # Plotting the target
    target_index = 0
    target_pos = [trajectory[0][target_index],trajectory[1][target_index]]
    target_id = p.createMultiBody(basePosition=[target_pos[0], target_pos[1], 0.01063783], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id_target)
    p.setRealTimeSimulation(1)

    # Real trajectory
    x = []
    y = []

    ################################################################################# Simulation
    while step != max_steps and target_index + 10 < trajectory[0].size:
            
        angles = Rotation.from_quat(robot.get_position_orientation()[1]).as_euler('zyx')[0]
        arc2 = np.arctan2(target_pos[1] - robot.get_position_orientation()[0][1], target_pos[0]-robot.get_position_orientation()[0][0])
        

        ############### Updated postions and errors
        x_pos = robot.get_position_orientation()[0][0] -  target_pos[0]
        y_pos = robot.get_position_orientation()[0][1] -  target_pos[1]

        # Recording the position to compare with the trajectory
        if (step % 10) == 0:
            x.append(robot.get_position_orientation()[0][0])
            y.append(robot.get_position_orientation()[0][1])
        

        # Correction of radians or angles
        if arc2 > 0:
            if angles < 0:
                angles = angles + 2*np.pi
        
        error_ang = arc2 - angles

        error = np.sqrt(np.power(x_pos,2) + np.power(y_pos,2))
        
        ################ Controller parameters
        # For angular
        kp= 0.10139 
        kd= 0.005 

        # For linear
        kp_ = 0.007535 
        kd_ = 0.00005 
        
        lin_vel = error * kp_ + (error-prev_error)*kd_
        ang_vel = kp * error_ang + (error_ang - prev_error_ang)*kd
        print(error_ang)

        print(ang_vel)
        command = [lin_vel,ang_vel]
        velocities = DifferentialDriveController._command_to_control(command, command, velocities) 
        joints = robot._joints.values()
        for joint, vel in zip(joints, velocities):
            joint.set_vel(vel)

        if error < 0.1: # Error threshold
            robot.apply_action([lin_vel, -ang_vel])
            # Plotting the target
            target_index += 10
            target_pos = [trajectory[0][target_index],trajectory[1][target_index]]
            p.removeBody(target_id)
            target_id = p.createMultiBody(basePosition=[target_pos[0], target_pos[1], 0.01063783], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id_target)


        prev_error = error
        prev_error_ang = error_ang

        step += 1
        s.step()
        

    ################################################################################# End of simulation
    plt.plot(targets[0], targets[1], 'o', label='Original Points')
    plt.plot(trajectory[0], trajectory[1], label='Cubic Spline')
    plt.plot(x,y, label='Real trajectory')
    plt.show()

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
