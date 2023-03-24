#%%
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
# import imread
from imageio import imread
from utils import depth_to_normal, normal_to_depth
# normal = imread('./out/guy_normal.png');
# depth = imread('./out/guy_depth.png');

# depth = double(depth) / 255.0;
# depth_avg = mean(depth, 3);

# depth_conv = depth_to_normal(depth_avg);
# imshow(depth_conv)

# python implementation of depth map to normal map
# normal = imread('./out/guy_normal.png')
depth = imread('./out/guy_depth.png')
# depth = imread('./out/0003_high_midas_v2_o2m.png')
# depth = imread('./out/0003_high_midas_v2_o2m.png')
# print(depth.shape)
# depth = depth.astype(np.float32) / np.max(depth)
depth = depth.astype(np.float32)
depth = depth / max(np.max(depth), 1e-6)
# depth_avg = np.mean(depth, axis=2)
# print(depth_avg.shape, depth.shape)
plt.imshow(depth)
# depth_avg = depth

def imshow(im):
    plt.figure()
    plt.imshow(im)



# %%
# normal = depth_to_normal(1./(depth))
normal = depth_to_normal(depth)
# print(depth)

# out[:, :, 0] = out[:, :, 0] / out_norm
# out[:, :, 1] = out[:, :, 1] / out_norm
# out[:, :, 2] = out[:, :, 2] / out_norm
imshow(normal)
normal = (normal + 1) / 2
imshow(normal)

# normal = imread('./out/metro_normal.png')
# normal = normal.astype(np.float32) / 255.0
# print(normal)
# print(normal.shape)
# unit_normal = normal / np.linalg.norm(normal, axis=2, keepdims=True)
# print(normalized_normal)
# pixel_norms = np.linalg.norm(unit_normal, axis=2)
# print(pixel_norms)
# imshow(normal)
# new_depth = normal_to_depth(normal, depth[0, 0])
# print(new_depth[0, 0], depth[0, 0])
# imshow(new_depth)


# %%
