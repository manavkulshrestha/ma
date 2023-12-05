# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file permits the agents to go to a same objective

import time as time_

import mujoco
import mujoco.viewer as viewer
from mujoco_mpc import agent as agent_lib
import numpy as np

import pathlib

# Creating model and data
m = mujoco.MjModel.from_xml_path("../../../../../mujoco_mpc/mjpc/tasks/quadruped/task_flat2.xml")
d = mujoco.MjData(m)

# Initializing our agent (agent server/executable)
agent = agent_lib.Agent(
        # This is to enable the ui
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat", 
        model=m)

agent2 = agent_lib.Agent(
        # Enable the ui and finds a freeport
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat", 
        model=m)

## Data for agents

# weights
agent.set_cost_weights({"Position": 0.15})
print("Cost weights:", agent.get_cost_weights())

# parameters
agent.set_task_parameter("Walk speed", 1.0)
print("Parameters:", agent.get_task_parameters())

# weights for second agent
agent2.set_cost_weights({"Position": 0.15})
print("Cost weights:", agent.get_cost_weights())

# parameters for second agent
agent2.set_task_parameter("Walk speed", 1.0)
print("Parameters:", agent.get_task_parameters())


# run planner for num_steps
num_steps = 8
for _ in range(num_steps):
    agent.planner_step()
    agent2.planner_step()

# Multiple goals
goals = [
    [5, 0, 0.26],
    [0, -2, 0.26],
    [-2, 0, 0.26],
    [0, -5, 0.26]
]
i = 0
goal_agent = []
goal_agent2 = []

# Settings different goals for the different robots
goal_agent = d.mocap_pos
goal_agent[0] = goals[0]
print(goal_agent)

goal_agent2 = d.mocap_pos
goal_agent2[0] = goals[1]
print(goal_agent)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.

    start = time_.time()
    while viewer.is_running() and time_.time():
        
        # set planner state
        agent.set_state(
            time=d.time,
            qpos=d.qpos[:19],
            qvel=d.qvel[:18],
            act=d.act,
            mocap_pos=goal_agent,
            mocap_quat=d.mocap_quat,
        )

        # set planner state for agent 2
        agent2.set_state(
            time=d.time,
            qpos=d.qpos[19:],
            qvel=d.qvel[18:],
            act=d.act,
            mocap_pos=goal_agent2,
            mocap_quat=d.mocap_quat,
        )

        print(d.mocap_pos) # Seems to take only in consideration the second agents goal or the changing goal
        # Obtain their actions
        agent_ctrl = agent.get_action()
        agent_ctrl2 = agent2.get_action()
        agent_ctrl = np.append(agent_ctrl, agent_ctrl2, axis=0)
        d.ctrl = agent_ctrl
        if (np.linalg.norm(d.mocap_pos[0] - d.body('trunk').xpos) < 1) or (np.linalg.norm(d.mocap_pos[0] - d.body('trunk2').xpos) < 1):
            print("\nARRIVED!")
            goal_agent[0] = goals[i]
            i = (i + 1) % len(goals)

        mujoco.mj_step(m,d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
