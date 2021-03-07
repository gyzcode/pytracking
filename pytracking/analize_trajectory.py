# Analize target trajectory.

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
from numpy import linalg as la
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
        std_x, std_y = .3, .3
        dt = 1.0

        kf = KalmanFilter(4, 2)
        kf.x = np.array([0., 0., 0., 0.])
        kf.R = np.diag([std_x**2, std_y**2])
        kf.F = np.array([[1, 0, 1, 0], 
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        # kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.0002)
        # kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.0002)
        kf.Q = np.identity(4) * 1e-4

        # Prepare for camera movement compensation
        warp_matrix = load_warp_matrix(seq.name)

        # Calculate ground truth center
        gt = np.array(seq.ground_truth_rect)
        gt_center = gt[:,0:2] + gt[:,2:4] / 2

        distances = []


        idx = 0
        for fn in seq.frames:
            image = cv.imread(fn)
    
            if idx > 0:

                # Camera movement compensation
                prev_pt = gt_center[idx-1]
                warp_pt = cv.perspectiveTransform(prev_pt.reshape(1,1,2), warp_matrix[idx-1]).reshape(-1)

                # Background (whole image) displacement
                bg_dist = warp_pt - prev_pt

                # calculate search center
                search_center = kf.x[2:4] + warp_pt

                distance = la.norm(gt_center[idx] - search_center)
                distances.append(distance)

                # Foreground (target) displacement
                fg_dist = gt_center[idx] - gt_center[idx-1]

                # Target absolute displacement
                abs_dist = fg_dist - bg_dist

                cv.circle(image, tuple(search_center.astype(int)), 3, (0,0,255), -1)

                measurement = kf.x[0:2] + abs_dist

                # Kalman filter predict and update
                kf.predict()
                
                kf.update(measurement)
                print(abs_dist)
                print(kf.x)
                print('----------------------')


            cv.circle(image, tuple(gt_center[idx].reshape(-1).astype(int)), 3, (0,255,0), -1)

            cv.imshow('test', image)
            key = cv.waitKey()
            if key == 27:
                break

            idx = idx + 1



    


if __name__ == '__main__':
    main()