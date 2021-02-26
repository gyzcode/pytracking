'''
Created on Feb 25, 2021
@author: Yuzhang Gu
'''
from kalman2d import Kalman2D

class Trajectory:
    """A class implementing one trajectory"""

    globID = 0
    predicted_max = 10
    iou_threshod = 0.1

    def __init__(self):
        self.id = Trajectory.globID
        Trajectory.globID += 1
        self.reset()

    def reset(self):
        self.points = []
        self.bbox = []
        self.predicted_count = 0
        self.kf2d = Kalman2D()

    def update(self, points, bbox):
        num = len(points)
        if num == 0:
            self.kf2d.update()
            self.points.append(self.kf2d.getPrediction())
            self.predicted_count = self.predicted_count + 1
        elif num == 1:
            self.kf2d.update(points[0], points[1])
            self.points.append(self.kf2d.getEstimate())
            self.predicted_count = 0
        elif num > 1:
            max_iou = 0
            max_iou_p = None
            for p in points:
                iou = 0 #待计算
                if max_iou < iou:
                    max_iou = iou
                    max_iou_p = p
            if max_iou > Trajectory.iou_threshod:
                self.kf2d.update(max_iou_p)
                self.points.append(max_iou_p)
                self.predicted_count = 0
            else:
                self.kf2d.update()
                self.points.append(self.kf2d.getPrediction())
                self.predicted_count = self.predicted_count + 1

        if self.predicted_count > Trajectory.predicted_max:
            self.reset()

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

