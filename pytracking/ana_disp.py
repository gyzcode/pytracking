# analize displacement

import os
import sys
import numpy as np
import cv2 as cv
from numpy.core.shape_base import vstack
from numpy.lib.npyio import savetxt

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.utils.load_warp_matrix import load_warp_matrix

dataset = get_dataset('otb')

total_hist = []
for i in range(1, 2):
    total_norm_dists = []
    for seq in dataset:
        warp_matrix = load_warp_matrix(seq.name, len(seq.frames)+1)
        gt = seq.ground_truth_rect
        centers = np.vstack([gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2]).transpose()
        searchs = np.linalg.norm(gt[:, 2:4], ord = 2, axis = 1, keepdims = True) * 2 * np.sqrt(2)
        if min(searchs) == 0:
            print('error')

        # Compensate camera movement
        compensate = True

        if compensate:
            old_centers = []
            for j in range(0, len(seq.frames)-i):
                old_pos = centers[j].reshape(1,1,2)
                new_pos = cv.perspectiveTransform(old_pos, warp_matrix[j])
                old_centers.append(new_pos.reshape(-1))
            disps = centers[i:] - old_centers
        else:
            disps = centers[i:] - centers[0: -i]

        dists = np.linalg.norm(disps, ord = 2, axis = 1, keepdims = True)
        norm_dists = dists / searchs[i:]
        if len(total_norm_dists) == 0:
            total_norm_dists = norm_dists
        else:
            total_norm_dists = vstack([total_norm_dists, norm_dists])
    hist = np.histogram(total_norm_dists, bins=10, range=(0,1))
    if len(total_hist) == 0:
        total_hist = hist[0]
    else:
        total_hist = vstack([total_hist, hist[0]])
savetxt('./hist.txt', total_hist, delimiter='\t', fmt='%d')

