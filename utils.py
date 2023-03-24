import numpy as np

def depth_to_normal(depth):
    z = 0.001
    print(depth[:, -1:].shape, depth[:, 1:].shape)
    dx = np.concatenate((depth[:, 1:], depth[:, -1:]), axis=1) - depth
    dy = np.concatenate((depth[1:, :], depth[-1:, :]), axis=0) - depth
    output = np.zeros((depth.shape[0], depth.shape[1], 3))
    grad_appended_one = np.concatenate((np.expand_dims(-dx, axis=2), np.expand_dims(-dy, axis=2), np.expand_dims(np.ones(depth.shape) * z, axis=2)), axis=2)
    grad_pixel_wise_norm = np.linalg.norm(grad_appended_one, axis=2)
    nx = -dx / grad_pixel_wise_norm
    ny = -dy / grad_pixel_wise_norm
    nz = z / grad_pixel_wise_norm
    # nx = -dx
    # ny = -dy
    # nz = z

    output[:, :, 0] = nx
    output[:, :, 1] = ny
    output[:, :, 2] = nz

    return output

def normal_to_depth(normal, initial):
    output = np.zeros((normal.shape[0], normal.shape[1]))
    output[0, 0] = initial
    for i in range(1, normal.shape[0]):
        output[i, 0] = output[i - 1, 0] - normal[i - 1, 0, 1]
    # fill all the columns
    for j in range(1, normal.shape[1]):
        output[:, j] = output[:, j - 1] - normal[:, j - 1, 0]
    return output