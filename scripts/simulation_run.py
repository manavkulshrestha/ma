from utils import sim

simulator = sim.Sim(xml_path="scripts/utils/xmls/unitree_go1/task_flat.xml")

simulator.view_sim(debug=True)