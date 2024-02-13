import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# Mujoco Libraries
import time
import mujoco
import mujoco.viewer
from scripts.sensors import Sensors
from vectorized_point_cloud import VectorizedPC

# Open3d Libraries
import open3d as o3d

def create_sim(xml_str, height, width, rgb=False, depth=False, segmentation=False):
    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)

    res = [m, d]

    if rgb:
        r_rgb = mujoco.Renderer(m, height, width)
        res.append(r_rgb)

    if depth:
        r_depth = mujoco.Renderer(m, height, width)
        r_depth.enable_depth_rendering()
        res.append(r_depth)

    if segmentation:
        r_seg = mujoco.Renderer(m, height, width)
        r_seg.enable_segmentation_rendering()
        res.append(r_seg)

    return res

def new_xml(models, quat):
    assets = ""
    geoms = ""

    for model in models:
        x = (np.random.random() - 0.5)
        y = (np.random.random() - 0.5) 
        z = (np.random.random() - 0.5) + 0.395
        assets += f'<mesh name="{model}"  file="{model}" scale="1 1 1"/>\n'
        geoms += f'<geom name="{model}" pos="{x} {y} {z}" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="{model}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>\n'

    xml =  f"""
    <mujoco>
        <asset>
            {assets}
        </asset>
        <worldbody>
            <light name="top" pos="0 0 1"/>
            {geoms}
                        
            <camera name="camera1" pos="0 -1 0.395" euler="90 0 0" mode="fixed" fovy="60" />
            <camera name="camera2" pos="1 0 0.395"  euler="90 90 0" mode="fixed" fovy="60" />
            <camera name="camera3" pos="0 1 0.395"  euler="90 180 0" mode="fixed" fovy="60" />
            <camera name="camera4" pos="-1 0 0.395" euler="90 270 0" mode="fixed" fovy="60" />

            <camera name="camera5" pos="0.75 -0.75 0.395"  euler="90 45 0" mode="fixed" fovy="60" />
            <camera name="camera6" pos="0.75 0.75 0.395"   euler="90 135 0" mode="fixed" fovy="60" />
            <camera name="camera7" pos="-0.75 0.75 0.395"  euler="90 225 0" mode="fixed" fovy="60" />
            <camera name="camera8" pos="-0.75 -0.75 0.395" euler="90 315 0" mode="fixed" fovy="60" />
              
        </worldbody>
    </mujoco>
    """

    return xml

def show_pc(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_plotly([pcd])


def frames(m, d, r_rgb, r_depth, r_seg):
    from scripts.sensors import Sensors
    from vectorized_point_cloud import VectorizedPC
    # Obtaining extrinsic details of the cameras
    cam_num = m.cam_user.shape[0]
    cam_pos = []
    cam_rot = []
    for i in range(cam_num):
        cam_pos.append(m.cam_pos[i])
        cam_rot.append(R.from_quat(m.cam(i).quat).as_matrix())
    
    # Instantiating the sensors and the pointcloud
    sensor = Sensors()
    point_cloud = VectorizedPC((480, 640), 60)
    
    # Making first step for the simulation
    mujoco.mj_step(m, d)
    
    # Get RGB, depth, and segmentation images
    depth = [*sensor.get_depth_image_matrices(m, d, r_depth).values()]
    segmn = [*sensor.get_segment_image_matrices(m, d, r_seg).values()]
    
    # Obtaining the pointclouds from the depth images segmented
    pc = {}
    cameras = [0,1]
    for i in cameras:
        rot = (R.from_matrix(cam_rot[i]).as_euler('xyz', degrees=True)[0] - 180) * -1
        rot = R.from_euler('xyz', [rot, 0, 90], degrees=True).as_matrix()
        aux = point_cloud.get_segmented_points(depth[i],
                                               segmn[i],
                                               rot,
                                               cam_pos[i])
        for key in aux.keys():
            if key == -1:
                continue
            try:
                pc[key] = np.concatenate((pc[key], aux[key]))
            except:
                pc[key] = aux[key]
    return pc

iter = 0
list_stl = []
t = 0
times = [0,0]

rootdir = "../mjcf/dataset_ycb/models/"
stl_file = None
for subdir, dirs, files in os.walk(rootdir):
    if subdir.endswith("/google_16k") and iter < 1:# 120   {quat[0]/5}
        new_subdir = subdir.replace("../mjcf/dataset_ycb/models/", "")
        new_subdir = new_subdir.replace("/google_16k", "")
        for file in files:
            if file.endswith(".stl"):
                stl_file = os.path.join(subdir, file)
                list_stl.append(stl_file)

# Generate xml for simulation with variables
point_cloud = VectorizedPC((480, 640), 60)
quat = np.random.random(4)
quat = quat / np.linalg.norm(quat)
models = list_stl
models = np.random.choice(models, size=4, replace=False)
xml = new_xml(models, quat)

# Instantiate the simulation
m, d, r_rgb, r_depth, r_seg = create_sim(xml, 480, 640, 
                                            rgb=True, 
                                            depth=True, 
                                            segmentation=True)

first = {}
last = {}

with mujoco.viewer.launch_passive(m, d) as viewer:
# Close the viewer automatically after 30 wall-seconds.

    start = time.time()
    while viewer.is_running() and time.time() - start < 2.1:
        step_start = time.time()
        # Moves the objects
        point_cloud.move_simple(t=t, func=lambda x: (x/10) * 0.2 + 0.05, axis=0, model=m)
        
        # Obtaining the pointclouds and their segmentation 
        last = frames(m, d, r_rgb, r_depth, r_seg)
        times[1] = time.time() - start
        t += 0.1

        # policy
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a contro l signal before stepping the physics.
        mujoco.mj_step(m, d)
        if time.time() - start <=  0.5:
            first = frames(m, d, r_rgb, r_depth, r_seg)
            times[0] = time.time() - start
            

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            time.sleep(0.01)
    # Gets the centroid for each frame
    centroid_pos_init = point_cloud.centroids(first)
    centroid_pos_fin = point_cloud.centroids(last)
    
    # Adding the velocity to the feature vector
    feature_vector = point_cloud.calc_vel(centroid_pos_fin, centroid_pos_init, times, m)
    
