#%%
import numpy as np
import matplotlib.pyplot as plt
# import imread
from imageio import imread
from depth_to_normal import depth_to_normalt
# normal = imread('./out/guy_normal.png');
# depth = imread('./out/guy_depth.png');

# depth = double(depth) / 255.0;
# depth_avg = mean(depth, 3);

# depth_conv = depth_to_normal(depth_avg);
# imshow(depth_conv)

# python implementation of depth map to normal map
# normal = imread('./out/guy_normal.png')
depth = imread('./out/guy_depth.png')
depth = depth.astype(np.float32) / 255.0
depth_avg = np.mean(depth, axis=2)

# %%
def depth_to_normal(depth):
    z = 0.02
    dx = np.concatenate((depth[:, 1:], depth[:, -1:]), axis=1) - depth
    dy = np.concatenate((depth[1:, :], depth[-1:, :]), axis=0) - depth
    output = np.zeros((depth.shape[0], depth.shape[1], 3))
    grad_appended_one = np.concatenate((np.expand_dims(-dx, axis=2), np.expand_dims(-dy, axis=2), np.expand_dims(np.ones(depth.shape) * z, axis=2)), axis=2)
    grad_norm = np.linalg.norm(grad_appended_one)
    nx = -dx / grad_norm
    ny = -dy / grad_norm
    nz = z / grad_norm

    output[:, :, 0] = nx
    output[:, :, 1] = ny
    output[:, :, 2] = nz
    # normalize each pixel to unit
    output = output / np.linalg.norm(output, axis=2, keepdims=True)
    # normalize the whole imge
    # output = output / np.linalg.norm(output)

    return output

out = depth_to_normal(depth_avg)
out_normal_color_map = (out + 0.5) / 2
plt.imshow(out_normal_color_map)

# %%

# %%

# %%

# %%
