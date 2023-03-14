#%%
import numpy as np
import matplotlib.pyplot as plt
# import imread
from imageio import imread
# normal = imread('./out/guy_normal.png');
# depth = imread('./out/guy_depth.png');

# depth = double(depth) / 255.0;
# depth_avg = mean(depth, 3);

# depth_conv = depth_to_normal(depth_avg);
# imshow(depth_conv)

# python implementation of depth map to normal map
# normal = imread('./out/guy_normal.png')
# depth = imread('./out/guy_depth.png')
depth = imread('./out/0003_high_midas_v2_o2m.png')
print(depth.shape)
depth = depth.astype(np.float32) / 255.0
# depth_avg = np.mean(depth, axis=2)
# print(depth_avg.shape, depth.shape)
plt.imshow(depth)
depth_avg = depth

def imshow(im):
    plt.figure()
    plt.imshow(im)


# %%
def depth_to_normal(depth):
    z = 1
    print(depth[:, -1:].shape, depth[:, 1:].shape)
    dx = np.concatenate((depth[:, 1:], depth[:, -1:]), axis=1) - depth
    dy = np.concatenate((depth[1:, :], depth[-1:, :]), axis=0) - depth
    output = np.zeros((depth.shape[0], depth.shape[1], 3))
    grad_appended_one = np.concatenate((np.expand_dims(-dx, axis=2), np.expand_dims(-dy, axis=2), np.expand_dims(np.ones(depth.shape) * z, axis=2)), axis=2)
    grad_pixel_wise_norm = np.linalg.norm(grad_appended_one, axis=2)
    nx = -dx / grad_pixel_wise_norm
    ny = -dy / grad_pixel_wise_norm
    nz = z / grad_pixel_wise_norm

    output[:, :, 0] = nx
    output[:, :, 1] = ny
    output[:, :, 2] = nz

    return output

normal = depth_to_normal(depth_avg)

# out[:, :, 0] = out[:, :, 0] / out_norm
# out[:, :, 1] = out[:, :, 1] / out_norm
# out[:, :, 2] = out[:, :, 2] / out_norm
imshow(depth)
normal = (normal + 1) / 2
imshow(normal)

# %%