import time
import mujoco as mj
import mujoco
import mujoco.viewer
import mujoco.renderer
import mujoco.glfw
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
from utility import unit
from controller import Controller
from human import Human

# notes: rolling friction may be too low, maybe should be velocity servo
# brownion motion and repulsion field for agent movement


def move_simple(name, *, t, func, axis=-1, model):
  model.site(name).pos[axis] = func(t)

def Drawcloudpoints(rgbd_image,cam_mat):
  # Creates Cloud points with default camera settings
  pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,cam_mat)

      # Adjusts the Point cloud
  pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

      # Shows
  o3d.visualization.draw_geometries([pcd])



def PinholeCameraIntrinsic(rgbd_image):
# Camera intrisic math -----
  fovy = math.radians(60)
      # Image
  img_width = np.asarray(rgbd_image.depth).shape[1] 
  img_height = np.asarray(rgbd_image.depth).shape[0]
  aspect_ratio = img_width/img_height
  fovx = 2 * math.atan(math.tan(fovy / 2) * aspect_ratio)

      # Focal length
      #( image. height / 2.0 ) / tan( (M_PI * FOV/180.0 )/2.0 )
  fy = img_height / (2 * math.tan(fovy / 2)* aspect_ratio)
  fx = img_width / (2 * math.tan(fovy / 2)* aspect_ratio)

      # Center
  cx = img_width/2
  cy = img_height/2  

  # PinholeCameraIntrinsic
  cam_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fx, cx, cy)
  print(cam_mat.intrinsic_matrix)
  
  return cam_mat




def rgbd_images(pixels_depth_cloud, original):
  # Turning image depth to gray scales and resize images
  img = cv2.cvtColor(pixels_depth_cloud, cv2.COLOR_BGR2RGB)   # BGR -> RGB
  img = cv2.resize(img,(640,480))
  o3d_depth = o3d.geometry.Image(np.ascontiguousarray(img).astype(np.float32))
      
  image=cv2.resize(original,(640,480))
  cv2.imwrite("yes.png", original)
  color = o3d.io.read_image("yes.png")

  # Generates its own rgbd image
  rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color,o3d_depth)

  # Shows both rgbd images
  """
      plt.subplot(1, 2, 1)
      plt.title("Grayscale image")
      plt.imshow(rgbd_image.color)
      plt.subplot(1 ,2 ,2)
      plt.imshow(rgbd_image.depth)
      plt.show()
      """

  return rgbd_image


def main():

  m = mj.MjModel.from_xml_path('./mjcf/tec.xml')
  d = mj.MjData(m)
  r = mujoco.Renderer(m, 480, 640) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480
  r2 = mujoco.Renderer(m, 480, 640) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480
  r3 = mujoco.Renderer(m, 480, 640) # RGB = 1080 x 1920  default =  900, 1200   Depth =  640 x 480

  frame = 0

  t = 0
  start_sim = d.time


  with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.

    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
      step_start = time.time()
      #move_simple('human1', t=t, func=lambda x: (np.sin(x)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      #move_simple('human2', t=t, func=lambda x: (np.sin(x*2)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      #move_simple('human3', t=t, func=lambda x: (np.sin(x*3)+1)/2 * 0.2 + 0.05, axis=-1, model=m)
      mujoco.mj_step(m, d)
      t += 0.1
      r3.update_scene(d, camera="camera2")
      original = r3.render()

  
      # Pick up changes to the physics state, apply perturbations, update options from GUI.
      viewer.sync()
      r.enable_depth_rendering()
      r2.enable_depth_rendering()

      r.update_scene(d, camera="camera",)
      r2.update_scene(d, camera="camera2",)


      depth_r2 = r2.render() # Gives 480, 640 as shape
      pixels_depth_cloud = depth_r2.astype(np.float32)
      pixels_depth_cloud[pixels_depth_cloud < 0.3] = np.nan
      pixels_depth_cloud[pixels_depth_cloud > 4.0] = np.nan
      
      

      rgbd_image = rgbd_images(pixels_depth_cloud, original)
      cam_mat = PinholeCameraIntrinsic(rgbd_image)
      Drawcloudpoints(rgbd_image, cam_mat)

      



      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
      time.sleep(0.01)
    r.disable_depth_rendering()
    r2.disable_depth_rendering()




if __name__ == '__main__':
  main()
