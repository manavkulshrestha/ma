from utils import sim

# Create a simulator from xml
simulator = sim.Sim(xml_path="scripts/utils/xmls/unitree_go1/task_flat.xml")

# Run the simulation
simulator.view_sim(debug=True)