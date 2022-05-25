"""
Simple experiment using GTSAM in a GraphSLAM context.

A series of
"""
from eurocreader.eurocreader import EurocReader
from graphslam.keyframemanager import KeyFrameManager
import numpy as np
from tools.homogeneousmatrix import HomogeneousMatrix
from tools.euler import Euler


def main():
    # Prepare data
    directory = '/media/arvc/INTENSO/DATASETS/dos_vueltas_long_range'
    euroc_read = EurocReader(directory=directory)
    scan_times, gt_pos, gt_orient = euroc_read.prepare_experimental_data(deltaxy=5.5, deltath=0.5,
                                                                         nmax_scans=None)
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times)
    for i in range(0, len(scan_times)):
        print('Iteration (keyframe): ', i)
        keyframe_manager.add_keyframe(i)
        keyframe_manager.keyframes[i].load_pointcloud()
        keyframe_manager.keyframes[i].pre_process()

        T = HomogeneousMatrix([0, 0, 0], Euler([0, 0, -np.pi/2]))
        #transform by a known transformaition
        keyframe_manager.keyframes[i].transform_by_T(T)

        keyframe_manager.add_keyframe(i+1)
        keyframe_manager.keyframes[i+1].load_pointcloud()
        keyframe_manager.keyframes[i+1].pre_process()

        # caution, need to use something with a prior
        itj, corr = keyframe_manager.compute_transformation_global(i, i+1, method='D')
        atb, rmse = keyframe_manager.compute_transformation_local(i, i+1, method='B', initial_transform=itj.array)
        # eabs = np.abs(T.t2v(n=3)-itj.t2v(n=3))
        # erel = np.abs(T.t2v(n=3)-atb.t2v(n=3))


if __name__ == "__main__":
    main()
