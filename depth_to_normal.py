import numpy as np

def depth_to_normalt(depth):
    # z = 0.001
    # dx = x+1 - x
    # dx = np.concatenate((depth[:, 1:], depth[:, -1:]), axis=1) - depth
    # dy = np.concatenate((depth[1:, :], depth[-1:, :]), axis=0) - depth
    print('hello')
    # output = np.zeros((depth.shape[0], depth.shape[1], 3))
    # grad_appended_one = np.concatenate((np.expand_dims(-dx, axis=2), np.expand_dims(-dy, axis=2), np.expand_dims(np.ones(depth.shape) * z, axis=2)), axis=2)
    # return output