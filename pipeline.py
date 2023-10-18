import open3d as o3d
import torch
from feature_detection import Net
import mujoco
import mujoco.renderer
from sensors import Sensors
from vectorized_point_cloud import VectorizedPC
from scipy.spatial.transform import Rotation as R
import numpy as np

# Pipeline execution of the model

def generate_random_rgb_values(n):
    """Generates n random equally spaced RGB values.

    Args:
        n: The number of RGB values to generate.

    Returns:
        A numpy array of shape (n, 3) containing the RGB values.
    """

    # Generate a random array of values between 0 and 255.
    rgb_values = np.random.randint(0, 256, size=(n, 3))

    # Normalize the values to be between 0 and 1.
    rgb_values = rgb_values / 255.0

    # Equally space the RGB values.
    rgb_values = rgb_values * (n - 1) / (n - 1)

    return rgb_values

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



def new_xml(model, quat):
    xml =  f"""
    <mujoco>
        <asset>
            <mesh name="{model}" file="ycb/{model}" scale="1 1 1"/>
        </asset>
        <worldbody>
            <light name="top" pos="0 0 1"/>
            <geom pos="0 0 0" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="{model}"
            quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
            />
                        
            <camera name="camera1" pos="-1 0.1 0" xyaxes="0 -0.5 0 -0.01 0 2" mode="fixed" fovy="60" />
            <camera name="camera2" pos="1 0.1 0" xyaxes="0 0.5 0 0.01 0 2" mode="fixed" fovy="60" />
            <camera name="camera3" pos="1 0.3 0.5" xyaxes="0 0.5 0 0.01 0 2" mode="fixed" fovy="60" />
            <camera name="camera4" pos="1 0.1 -0.2" xyaxes="0 0.5 0 0.01 0 2" mode="fixed" fovy="60" />
              
        </worldbody>
    </mujoco>
    """

    return xml

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
    xml = """
    <>
    </>
    """

    xml = new_xml('003_cracker_box.stl', [0, 0, 0, 1])

    # Instantiate the simulation
    m, d, r_rgb, r_depth, r_seg = create_sim(xml, 640, 480, 
                                             rgb=True, 
                                             depth=True, 
                                             segmentation=True)
    
    # Obtaining extrinsic details of the cameras
    cam_num = m.cam_user.shape[0]
    cam_pos = []
    cam_rot = []
    for i in range(cam_num):
        cam_pos.append(m.cam_pos[i])
        cam_rot.append(R.from_quat(m.cam(i).quat).as_matrix())
    
    # Instantiating the sensors and the pointcloud
    sensor = Sensors()
    point_cloud = VectorizedPC((640, 480), 60)
    
    # Making first step for the simulation
    mujoco.mj_step(m, d)

    # Get RGB, depth, and segmentation images
    depth = [*sensor.get_depth_image_matrices(m, d, r_depth).values()]
    segmn = [*sensor.get_segment_image_matrices(m, d, r_seg).values()]

    # Obtaining the pointclouds from the depth images segmented
    pc = {}
    for i in range(cam_num):
        cam_pc = point_cloud.get_segmented_points(depth[i],
                                                  segmn[i],
                                                  cam_rot[i],
                                                  cam_pos[i])
        for key in list(cam_pc.keys()):
            if key in pc:
                pc[key] = np.append(pc[key], cam_pc[key], axis=0)
            else:
                pc[key] = cam_pc[key]

    # Generating colors
    keys = list(pc.keys())
    # colors = generate_random_rgb_values(len(keys))

    # Displaying all pointclouds in a single 3d window
    pcd = o3d.geometry.PointCloud()
    for i in range(len(keys)):
        pcd.points = o3d.utility.Vector3dVector(pc[keys[i]])
        # pcd.colors = o3d.utility.Vector3dVector(colors[i])

    o3d.visualization.draw_geometries([pcd])

