import os
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import mujoco.renderer
import mujoco.glfw
import numpy as np
from scripts.sensors import Sensors
from point_cloud import PointCloud
import random
from scipy.spatial import distance



# ------------ Model Reading -------------

model_names = os.listdir("ycb")


# ---------- Helper Functions ------------

# Function to generate new xml
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


# Function to get delete outlier points
def erase_outliers(point_cloud, threshold=0.5):

    # Calculate the mean and standard deviation of the point cloud
    mean = np.mean(point_cloud, axis=0)
    std_dev = np.std(point_cloud, axis=0)

    # Calculate the Mahalanobis distance for each point from the mean
    mahalanobis_distances = distance.cdist(point_cloud, [mean], 'mahalanobis', VI=np.linalg.inv(np.diag(std_dev)))

    # Create a mask to identify outliers
    outlier_mask = np.squeeze(mahalanobis_distances > threshold)
    
    # Filter out the outliers from the point cloud
    filtered_point_cloud = point_cloud[~outlier_mask]

    return filtered_point_cloud


# Function to get n random elements from an array with probabilities
def get_elements_with_probability(arr=[0,1,2,3]):
    # Define the probabilities
    probabilities = [0.7, 0.25, 0.05]

    # Determine the number of elements to select based on probabilities
    num_elements = random.choices([1, 2, 3], probabilities)[0]

    # Shuffle the array to randomize the selection
    random.shuffle(arr)

    # Return a set of the selected elements
    return arr[:num_elements]



# -------------- Main Loop ---------------

num_data = 100_000

# Creating instance of classes for data collection
sensors = Sensors()
point_c = PointCloud((480, 640), 60, downsample=3)

# Constant values
num_cameras = 4
master_m = mujoco.MjModel.from_xml_string(
                        new_xml("003_cracker_box/textured.obj", 
                        np.array([1,0,0,0])))
poses = []
rot_matrices = []
for i in range(num_cameras):
    # Get the position of the camera
    pos = master_m.cam_pos[i]
    poses.append(pos)

    # Get the rotation of the camera
    rot_vect = R.from_quat(master_m.cam(i).quat).as_matrix()
    rot_matrices.append(rot_vect)

for model in model_names:
    name = model.split(".")[0]
    for n in range(100_000 // len(model_names)):
        # Generating random quaternion
        quat = np.random.random(4)
        quat = quat / np.linalg.norm(quat)

        # Generating new xml
        xml = new_xml(model, quat)

        # Create the model and data objects.
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)

        # Make renderer, render and show the pixels
        depth = mujoco.Renderer(m,480, 640); depth.enable_depth_rendering()
        sgmnt = mujoco.Renderer(m,480, 640); sgmnt.enable_segmentation_rendering()
        
        # Doing initial loading
        mujoco.mj_step(m, d)

        # This can be obtained from the model at the beginning of the simulation
        cameras = get_elements_with_probability()

        # Renderer with depth enabled
        depth_frames = [*sensors.get_depth_image_matrices(m, d, depth, cameras).values()]

        # Renderer with segmentation enabled
        sgmnt_frames = [*sensors.get_segment_image_matrices(m, d, sgmnt, cameras).values()]
        
        # Get the point cloud from the segmented map
        points = [*point_c.get_segmented_map(depth_frames, sgmnt_frames, 
                                             [poses[i] for i in cameras], 
                                             [rot_matrices[i] for i in cameras], 
                                             m).values()]

        points = np.array(points[0])

        # Save the point cloud
        np.save(f'dataset_pc/{name}_{"%06d" % (n+1,)}.npy', points[0])
    