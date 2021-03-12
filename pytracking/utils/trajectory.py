'''
Created on Feb 25, 2021
@author: Yuzhang Gu
'''

import os
import sys

from numpy.core.numerictypes import maximum_sctype

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.kalman2d import Kalman2D
import torch
import copy
from pytracking.analysis import calc_iou_overlap

class Trajectory:
    """A class implementing one trajectory"""

    globID = 0
    predicted_max = 10
    iou_threshod = 0.1

    def __init__(self, pos):
        self.id = Trajectory.globID
        Trajectory.globID += 1

        self.points = []
        self.bbox = []
        self.predicted_count = 0
        self.kf2d = Kalman2D()
        self.flag = ''

        self.pos = pos
        self.old_pos = None
        self.delta = []
        self.dist = []
        self.delta_mean = 0
        self.delta_stdev = 0
        self.updated = False


    def update(self, points):
        nearest = 0
        num = len(points)
        if num == 0:
            self.kf2d.update1()
            self.points.append(torch.tensor(self.kf2d.getPrediction()))
            self.predicted_count = self.predicted_count + 1
            self.flag = 'predicted'
        elif num == 1:
            self.kf2d.update(points[0])
            self.points.append(torch.tensor(self.kf2d.getEstimate()))
            self.predicted_count = 0
            self.flag = 'normal'
        elif num > 1:
            min_dist = 1e5
            min_dist_p = None
            i = 0
            for p in points:
                delta = torch.tensor(p) - torch.tensor(self.points[-1])
                dist = torch.norm(delta)
                if min_dist > dist:
                    min_dist = dist
                    min_dist_p = p
                    nearest = i
                i = i+1
            if min_dist < torch.max(self.bbox[:, 2:4]):
                self.kf2d.update(min_dist_p)
                self.points.append(torch.tensor(self.kf2d.getEstimate()))
                self.predicted_count = 0
                self.flag = 'normal'
            else:
                self.kf2d.update1()
                self.points.append(torch.tensor(self.kf2d.getPrediction()))
                self.predicted_count = self.predicted_count + 1
                self.flag = 'predicted'

        if self.predicted_count > Trajectory.predicted_max:
            self.reset()

        return nearest

    def getPoints(self):
        return self.points

    @staticmethod
    def decGlobID():
        Trajectory.globID -= 1

    @staticmethod
    def resetGlobID():
        Trajectory.globID = 0

    def __str__(self):
        str  = "=== Trajectory ===\n"
        for p in self.points:
            str += repr(p) + ", "
        str += "\n"
        return str

    def __len__(self):
        return len(self.points)

