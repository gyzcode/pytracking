import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import time    
import cv2 as cv
import numpy as np
from visdom import Visdom
from pytracking.evaluation import get_dataset
from pytracking.evaluation import Tracker


class Analizer(Tracker):
    def __init__(self):
        self.pause_mode = False
        self.next = False
        self.prev = False
        

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode
            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.next = True
            elif data['key'] == 'ArrowLeft' and self.pause_mode:
                self.prev = True

    def analize(self):
        dataset = get_dataset('otb')
        dataset = [dataset['Basketball']]

        bbox = np.loadtxt('/home/gyz/workzone/pytracking/pytracking/tracking_results/dimp/dimp50/Basketball.txt')
        score_map = np.loadtxt('/home/gyz/workzone/pytracking/pytracking/tracking_results/dimp/dimp50/Basketball_score_map.txt')
        
        vis = Visdom(env = 'img')
        vis.register_event_handler(self._visdom_ui_handler, 'display')
        
        for seq in dataset:
            frame_num = 1
            # for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            while True:
                while True:
                    if not self.pause_mode:
                        frame_num += 1
                        break
                    elif self.prev:
                        frame_num -= 1
                        self.prev = False
                        break
                    elif self.next:
                        frame_num += 1
                        self.next = False
                        break
                    else:
                        time.sleep(0.1)

                image = self._read_image(seq.frames[frame_num])  
                vis.image(image.transpose(2, 0, 1), 'display')


def main():
    ana = Analizer()
    ana.analize()


if __name__ == '__main__':
    main()
