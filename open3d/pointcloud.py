import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# Se re-size of images
image = cv2.imread("myimgr_19.png") # Depth image in black and white from the camera
image=cv2.resize(image,(640,480))
cv2.imwrite("myimgr_19.png",image)

image = cv2.imread("myimgr_1.png") # Normal image of camera
image=cv2.resize(image,(640,480))
cv2.imwrite("myimgr_1.png",image)
# print(image.shape) # (480, 640, 3)

# Reading of images by open3d
color_raw = o3d.io.read_image("myimgr_1.png") # Color image
depth_raw = o3d.io.read_image("myimgr_19.png") # Gray depth image

# Creates an RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

# Plots the RGBD images
plt.subplot(1, 2, 1)
plt.title("Redwood grayscale image")
plt.imshow(rgbd_image.color)
plt.subplot(1 ,2 ,2)
plt.imshow(rgbd_image.depth)
plt.show()

# Creates the point cloud taking in consideration the camera standard measurements.
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                     o3d.camera.PinholeCameraIntrinsic(
                                                        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Rotates the point cloud
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

# Draws the Point clouds
o3d.visualization.draw_geometries([pcd])
