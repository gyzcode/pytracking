# -*- coding: utf-8 -*-

'''
Created on Feb 26, 2021
@author: Yuzhang Gu
'''

import cv2 as cv
 
class Kalman2D(object):
    '''
    A class for 2D Kalman filtering
    '''
 
    def __init__(self, processNoiseCov=1e-5, measurementNoiseCov=1, errorCovPost=1):
        '''
        Constructs a new Kalman2D object.  
        For explanation of the error covariances see
        http://en.wikipedia.org/wiki/Kalman_filter
        '''

        # 状态空间：位置--2d,速度--2d
        self.kf = cv.KalmanFilter(4, 2)

        self.kf.transitionMatrix = cv.setIdentity(self.kf.transitionMatrix)
        #加入速度 x = x + vx, y = y + vy
        # 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1
        #如果把下面两句注释掉，那么位置跟踪kalman滤波器的状态模型就是没有使用速度信息
        self.kf.transitionMatrix[0, 2] = 1
        self.kf.transitionMatrix[1, 3] = 1

        #初始化单位矩阵
        self.kf.measurementMatrix = cv.setIdentity(self.kf.measurementMatrix)

        #初始化带尺度的单位矩阵
        self.kf.processNoiseCov = cv.setIdentity(self.kf.processNoiseCov, processNoiseCov)
        self.kf.measurementNoiseCov = cv.setIdentity(self.kf.measurementNoiseCov, measurementNoiseCov)
        self.kf.errorCovPost = cv.setIdentity(self.kf.errorCovPost, errorCovPost)
 
        self.predicted = None
        self.estimated = None
 
    def update(self, point):
        '''
        Updates the filter with a new X,Y measurement
        '''
 
        self.measurement = point

        self.predicted = self.kf.predict()
        self.estimated = self.kf.correct(self.measurement)

    def update1(self):
        '''
        Updates the filter without any measurement
        '''
 
        self.predicted = self.kf.predict()

 
    def getEstimate(self):
        '''
        Returns the current X,Y estimate.
        '''
 
        return self.estimated[0:2]
 
    def getPrediction(self):
        '''
        Returns the current X,Y prediction.
        '''
 
        return [self.predicted[0,0], self.predicted[1,0]]
