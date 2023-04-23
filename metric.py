import numpy as np
from skimage.segmentation import slic
from imageio import imread
import sys


def compute_d3r(pred, target, freq_threshold, threshold, nsamples, debug=False, mask=None):
    """ Computing D3R metric using diff instead of ratio to compute the ordinal relations

    Args:
        gtdisp ([torch array]): Ground truth disparity map between 0,1 not containing Nan values. 
        preddisp ([torch array]): Prediction disparity map between 0,1 not containing Nan values. 
        freq_threshold ([float]): A threshold to define high frequecy changes
        threshold ([float]): Threshold to define ordinal relations based on diff
        nsamples ([int]): Number of superpixels created by SLIC alghorithm

    Returns:
        [float, list of tuples, list of tuples, 2darray]: computed error value, list of selected point pairs,
         list of point pairs that had mismatching orders, position of centers of superpixels
    """

    EPSILON = 1e-6
    
    gtdisp = target
    preddisp = pred
    mask = mask if mask is not None else np.ones_like(preddisp)

    centers, point_pairs = extract_pointpairs(gtdisp = gtdisp, nsamples = nsamples)

    not_matching_pairs = []
    selected_pairs = []
    d3r_error = 0
    for point_pair in point_pairs:
        center1_loc = np.floor(centers[point_pair[0]-1,:]).astype(np.int32)
        center2_loc = np.floor(centers[point_pair[1]-1,:]).astype(np.int32)

        mask_loc1 = mask[center1_loc[0],center1_loc[1]]
        mask_loc2 = mask[center2_loc[0],center2_loc[1]]
        if mask_loc1 == False or mask_loc2 == False:
            continue
        
        # only consider points with a high freq in ground truth for evaluation
        gt_freq_ord = compute_ordering(base = gtdisp, point_loc1 = center1_loc, point_loc2 = center2_loc, threshold = freq_threshold)
        if gt_freq_ord == 0:
            continue
        
        selected_pairs.append(point_pair)
        gt_ordering = compute_ordering(base = gtdisp, point_loc1 = center1_loc, point_loc2 = center2_loc, threshold = threshold)
        pred_ordering = compute_ordering(base = preddisp, point_loc1 = center1_loc, point_loc2 = center2_loc, threshold = threshold)

        if gt_ordering != pred_ordering:
            not_matching_pairs.append(point_pair)
        
        d3r_error += abs(gt_ordering - pred_ordering)

    d3r_error = d3r_error / (len(selected_pairs) + EPSILON)

    if debug:
        return d3r_error, selected_pairs, not_matching_pairs, centers
    else:
        return d3r_error

def compute_ordering(base, point_loc1, point_loc2, threshold):
    diff = base[point_loc1[0],point_loc1[1]] -  base[point_loc2[0],point_loc2[1]]
    if diff > threshold:
        ord = +1
    elif diff < -threshold:
        ord = -1
    else:
        ord = 0

    return ord

def extract_pointpairs(gtdisp, nsamples):
    segments = slic(gtdisp, n_segments=nsamples, compactness=1)
    segments_ids = np.unique(segments)

    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    point_pairs = []
    for i in range(bneighbors.shape[1]):
        if bneighbors[0][i] != bneighbors[1][i]:
            point_pairs.append((bneighbors[0][i],bneighbors[1][i]))
    
    return centers, point_pairs



pred = imread(sys.argv[1])
target = imread(sys.argv[2])

pred = pred.astype(np.float32) / 255
target = target.astype(np.float32) / 255
print(pred.shape)
print(target.shape)

res = compute_d3r(pred, target, freq_threshold=0.04, threshold=0.01, nsamples=4000, debug=False, mask=None)
print(res)