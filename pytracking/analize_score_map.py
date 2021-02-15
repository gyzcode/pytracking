import os
import sys
from numpy.core.fromnumeric import argmax

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import torch
import time    
import cv2 as cv
import numpy as np
from visdom import Visdom
from pytracking.evaluation import get_dataset
from pytracking.evaluation import Tracker
from pytracking.analysis import calc_iou_overlap


class Analizer(Tracker):
    def __init__(self):
        self.pause_mode = False
        self.next = False
        self.prev = False
        self.search = False
        

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode
            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.next = True
            elif data['key'] == 'ArrowLeft' and self.pause_mode:
                self.prev = True
            elif data['key'] == 's' and self.pause_mode:
                self.search = True

    def analize(self):
        dataset = get_dataset('otb')
        dataset = [dataset['Basketball']]
        dataset[0].ground_truth_rect = dataset[0].ground_truth_rect.astype(int)
        anno_bb = torch.tensor(dataset[0].ground_truth_rect)

        bbox = np.loadtxt('/home/gyz/workzone/pytracking/pytracking/tracking_results/dimp/dimp50/Basketball.txt')
        pred_bb = torch.tensor(bbox, dtype=torch.float64)
        bbox = bbox.astype(int)
        score_map = np.loadtxt('/home/gyz/workzone/pytracking/pytracking/tracking_results/dimp/dimp50/Basketball_score_map.txt')
        score_map = score_map.reshape(-1, 19, 19)
        for i in range(score_map.shape[0]):
            score_map[i, ...] = np.flipud(score_map[i, ...])

        iou = calc_iou_overlap(pred_bb, anno_bb)
        
        viz = Visdom(env = 'analize')
        viz.register_event_handler(self._visdom_ui_handler, 'frame')
        
        for seq in dataset:
            idx = 0
            while True:
                while True:
                    if not self.pause_mode:
                        idx += 1
                        break
                    elif self.prev:
                        idx -= 1
                        self.prev = False
                        break
                    elif self.next:
                        idx += 1
                        self.next = False
                        break
                    elif self.search:
                        idx += 1
                        if iou[idx] < 0.1:
                            self.search = False
                        break
                    else:
                        time.sleep(0.1)
                
                if not self.search:
                    image = self._read_image(seq.frames[idx])
                    cv.rectangle(image, (bbox[idx][0], bbox[idx][1]), (bbox[idx][0] + bbox[idx][2], bbox[idx][1] + bbox[idx][3]), (255, 0, 0), 2)
                    cv.rectangle(image, (seq.ground_truth_rect[idx][0], seq.ground_truth_rect[idx][1]), (seq.ground_truth_rect[idx][0] + seq.ground_truth_rect[idx][2], seq.ground_truth_rect[idx][1] + seq.ground_truth_rect[idx][3]), (0, 255, 0), 2)
                    viz.image(image.transpose(2, 0, 1), 'frame')
                    info_text = 'frame: {}<br>iou: {:.2f}<br>'.format(idx, iou[idx])
                    viz.text(info_text, 'info')
                    if (idx > 0):
                        viz.heatmap(score_map[idx - 1], 'score')
                        m = argmax(score_map[idx - 1])
                        x = m % 19
                        y = m // 19
                        info_text += 'max_loc: {},{}<br>'.format(x, y)
                    viz.text(info_text, 'info')



def main():
    ana = Analizer()
    ana.analize()


if __name__ == '__main__':
    main()
