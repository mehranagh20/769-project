import numpy as np
import matplotlib.pyplot as plt
# import imread
from imageio import imread
from PIL import Image
from numpy import asarray
from matplotlib import cm
import cv2

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import sys
depth = imread('./speaker_rgb.png_depth_grey_ours.png')
print(depth.shape)

color = Image.open('./speaker_rgb.png')
print(color.size)
color_image = asarray(color)
print(color_image.shape)

color = o3d.geometry.Image(color_image.astype(np.uint8))


depth = np.asarray(depth).astype(np.float32)
depth=depth+450
depth = o3d.geometry.Image(depth)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
W,H,_=color_image.shape
fx= 1.0471975511965976
fy=0.8172757101951849

cx=W/2
cy=H/2
intrinsic = o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
#intrinsic_path=('https://github.com/mehranagh20/769-project/blob/test/new-data-out/speaker_camera.json')
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
print(rgbd)

pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

o3d.visualization.draw_geometries([pcd])
#o3d.io.write_point_cloud('speaker_pc', pcd)