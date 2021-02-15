import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation import Tracker
from pytracking.analysis import playback_results


def main():
    trackers = [Tracker('dimp', 'dimp50', run_id=None)]

    dataset = get_dataset('otb')
    dataset = [dataset['Basketball']]

    for seq in dataset:
        playback_results(trackers, seq)
    


if __name__ == '__main__':
    main()
