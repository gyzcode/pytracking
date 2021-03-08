# Analize target trajectory.

from math import dist
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import argparse
import torch
import time    
import cv2 as cv
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from pytracking.evaluation import get_dataset
from pytracking.utils.load_warp_matrix import load_warp_matrix



def main():
    parser = argparse.ArgumentParser(description='Analize trajectory.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    args = parser.parse_args()

    # Load dataset
    dataset = get_dataset(args.dataset_name)
    if args.sequence != None:
        dataset = [dataset[args.sequence]]

    for seq in dataset:
        
        # Initialize kalman filter
        std_x, std_y = 0.1, 0.1
        dt = 1.0

        kf = KalmanFilter(2, 2)
        kf.x = np.array([0., 0.])
        kf.R = np.diag([std_x**2, std_y**2])
        kf.F = np.array([[1, 0], 
                        [0, 1]])
        kf.H = np.array([[1, 0],
                        [0, 1]])
        
        # kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=2)
        # kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=2)
        kf.Q = np.identity(2) * 1e-4

        # Prepare for camera movement compensation
        warp_matrix = load_warp_matrix(seq.name)

        # Calculate ground truth center
        gt = np.array(seq.ground_truth_rect)
        gt_center = gt[:,0:2] + gt[:,2:4] / 2

        distances = []

        idx = 0
        old_pos = gt_center[0]
        new_pos = None
        for fn in seq.frames:
            image = cv.imread(fn)
    
            if idx > 0:

                # Camera movement compensation
                prev_pt = old_pos
                warp_pt = cv.perspectiveTransform(prev_pt.reshape(1,1,2), warp_matrix[idx-1]).reshape(-1)

                # calculate search center
                search_center = kf.x + warp_pt
                cv.circle(image, tuple(search_center.astype(int)), 3, (0,0,255), -1)

                # Target absolute displacement
                abs_dist = gt_center[idx] - search_center
                
                distance = np.linalg.norm(abs_dist)
                distances.append(distance)
                if len(distances) > 20:
                    distances.pop(0)

                cv.circle(image, tuple(gt_center[idx].reshape(-1).astype(int)), 3, (0,255,0), -1)

                found = True
                cv.imshow('test', image)
                key = cv.waitKey()
                if key == 27:
                    break
                elif key == 102:    #'f'
                    found = False

                

                # Kalman filter predict and update
                kf.predict()
                
                if found:
                    kf.update(abs_dist)
                    new_pos = search_center + kf.x
                else:
                    new_pos = search_center
                
                old_pos = new_pos

                print(kf.x)

            idx = idx + 1



    


if __name__ == '__main__':
    main()