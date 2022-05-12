import numpy as np
from graphslam.keyframemanager import KeyFrame


def parse_log_file(filename):
    keyframes = []
    odometry = []
    with open(filename) as f:
        lines = f.readlines()
    i = 0
    for line in lines:
        kf = KeyFrame()
        if kf.fromstring(line):
            print("Reading Keyframe FLASER", i)
            keyframes.append(kf)
            odometry.append(kf.odometry)
            i += 1
    if len(keyframes) == 0:
        print('Could not read data from file')
        raise Exception
    return keyframes, np.array(odometry)
