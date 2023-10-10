import open3d as o3d
import torch
from feature_detection import Net
import mujoco
import mujoco.renderer
from sensors import Sensors

# Pipeline execution of the model

# #
# INITIALIZATION OF SIMULATION
# #
def create_sim(xml_str, width, height, rgb=False, depth=False, segmentation=False):
    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)

    res = [m, d]

    if rgb:
        r_rgb = mujoco.Renderer(m, height, width)
        res.append(r_rgb)

    if depth:
        r_depth = mujoco.Renderer(m, height, width)
        r_seg = mujoco.Renderer(m, height, width)
        res.append(r_depth)

    if segmentation:
        r_depth.enable_depth_rendering()
        r_seg.enable_segmentation_rendering()
        res.append(r_seg)

    return res

# # 
# INSTANTIATION OF THE MODEL
# #

# Load the model
MODEL_PATH = 'model.pt'



def load_model(model_path:str) -> Net:
    """
    Loads the model from the specified path directly in
    evaluation mode for inference.
    
    Parameters
    ----------
    model_path : str
        The path to the model to be loaded
    """
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return 



# Main loop
model = load_model(MODEL_PATH)



while True:
    # Generate xml for simulation with variables
    xml = "<></>"

    # Instantiate the simulation
    m, d, r_rgb, r_depth, r_seg = create_sim(xml, 640, 480, 
                                             rgb=True, 
                                             depth=True, 
                                             segmentation=True)
    sensor = Sensors()
    
    # Making first step for the simulation
    mujoco.mj_step(m, d)

    # Get RGB, depth, and segmentation images
    readings = sensor.get_depth_image_matrices(m, d, r_depth)

    # Joining all point clouds
    keys = list(readings.keys())

    for key in keys:
        readings[key] = sensor.get_points(readings[key])

    # Displaying all pointclouds in a single 3d window
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(readings['depth'])
    o3d.io.write_point_cloud("test.ply", pcd)

    pcd_load = o3d.io.read_point_cloud("test.ply")
    o3d.visualization.draw_geometries([pcd_load])

