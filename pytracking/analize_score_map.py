import os
import sys
from numpy.core.fromnumeric import argmax

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import argparse
import torch
import time    
import cv2 as cv
import numpy as np
from visdom import Visdom
from pytracking.evaluation import get_dataset
from pytracking.evaluation import Tracker
from pytracking.analysis import calc_iou_overlap


class Analizer():
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

    def analize(self, tracker_name, tracker_param, dataset_name, sequence, run_id):
        trackers = [Tracker(tracker_name, tracker_param, run_id)]

        dataset = get_dataset(dataset_name)
        if sequence != None:
            dataset = [dataset[sequence]]
        
        viz = Visdom(env = 'analize')
        viz.register_event_handler(self._visdom_ui_handler, 'frame')

        path = '/home/gyz/workzone/pytracking/pytracking/tracking_results/{}/{}_{:03d}/'.format(tracker_name, tracker_param, run_id)


        
        for seq in dataset:

            # ignore = True

            for tracker in trackers:
                # if ignore:
                #     if seq.name == 'KiteSurf':
                #         ignore = False
                #     else:
                #         continue

                seq.ground_truth_rect = seq.ground_truth_rect.astype(int)
                anno_bb = torch.tensor(seq.ground_truth_rect)

                bbox = np.loadtxt(path + '{}.txt'.format(seq.name))
                pred_bb = torch.tensor(bbox, dtype=torch.float64)
                bbox = bbox.astype(int)
                score_map = np.loadtxt(path + '{}_score_map.txt'.format(seq.name))
                if tracker_param == 'dimp50':
                    map_size = 19
                else:
                    map_size = 23
                score_map = score_map.reshape(-1, map_size, map_size)

                for i in range(score_map.shape[0]):
                    score_map[i, ...] = np.flipud(score_map[i, ...])

                iou = calc_iou_overlap(pred_bb, anno_bb)

                idx = 0
                while idx < len(seq.frames) - 1:
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
                            time.sleep(0.2)
                    
                    if not self.search:
                        image = tracker._read_image(seq.frames[idx])
                        cv.rectangle(image, (bbox[idx][0], bbox[idx][1]), (bbox[idx][0] + bbox[idx][2], bbox[idx][1] + bbox[idx][3]), (255, 0, 0), 2)
                        cv.rectangle(image, (seq.ground_truth_rect[idx][0], seq.ground_truth_rect[idx][1]), (seq.ground_truth_rect[idx][0] + seq.ground_truth_rect[idx][2], seq.ground_truth_rect[idx][1] + seq.ground_truth_rect[idx][3]), (0, 255, 0), 2)
                        viz.image(image.transpose(2, 0, 1), 'frame')
                        info_text = 'seq: {}<br>frame: {}<br>iou: {:.2f}<br>'.format(seq.name, idx, iou[idx])
                        viz.text(info_text, 'info')
                        if (idx > 0):
                            viz.heatmap(score_map[idx - 1], 'score')
                            m0 = score_map[idx - 1].max()
                            m = argmax(score_map[idx - 1])
                            x = m % map_size
                            y = m // map_size
                            info_text += 'max: {:.3f}<br>max_loc: {},{}<br>'.format(m0, x, y)
                        viz.text(info_text, 'info')



def main():
    parser = argparse.ArgumentParser(description='Analize score map.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')

    args = parser.parse_args()
    
    ana = Analizer()
    ana.analize(args.tracker_name, args.tracker_param, args.dataset_name, args.sequence, args.runid)


if __name__ == '__main__':
    main()
