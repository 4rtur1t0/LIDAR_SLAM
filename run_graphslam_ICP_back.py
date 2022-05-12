"""
Simple experiment using GTSAM in a GraphSLAM context.

A series of
"""
from eurocreader.bagreader import RosbagReader
from graphslam.dataassociation import DataAssociation
from graphslam.graphslam import GraphSLAM
import gtsam

from graphslam.keyframemanager import KeyFrameManager

SIGMAS_ICP = gtsam.Point3(0.1, 0.1, 0.06)
SIGMAS_PRIOR = gtsam.Point3(0.05, 0.05, 0.01)
ICP_NOISE = gtsam.noiseModel.Diagonal.Sigmas(sigmas=SIGMAS_ICP)
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(sigmas=SIGMAS_PRIOR)


def main():
    # create the graphslam graph
    graphslam = GraphSLAM(icp_noise=ICP_NOISE, prior_noise=PRIOR_NOISE)
    # create the Data Association object
    dassoc = DataAssociation(graphslam, delta_index=80, xi2_th=20.0)
    # reads data from the experiment
    rosbag = RosbagReader(filename='realdata/husky_playpen_1loop.bag',
                          odo_topic='/odometry/filtered',
                          points_topic='/points')
    rosbag.read_rosbag_data3D(deltaxy=0.5, deltath=0.2)
    # get initial observation
    odometry, points = rosbag.get_observation()
    odo_gt = []
    odo_gt.append(odometry)
    # create keyframemanager and add initial observation
    keyframe_manager = KeyFrameManager()
    keyframe_manager.add_keyframe(odometry=odometry, points=points)

    for i in range(1, len(rosbag.odometry)):
        odometry, points = rosbag.get_observation()
        odo_gt.append(odometry)
        # CAUTION: odometry is not used. ICP computed without any prior
        keyframe_manager.add_keyframe(odometry=odometry, points=points)
        # compute relative motion between scan i and scan i-1 0 1, 1 2...
        atb = keyframe_manager.compute_transformation_local(i-1, i)
        # consecutive edges. Adds a new node AND EDGE with restriction aTb
        graphslam.add_consecutive_observation(atb)
        # non-consecutive edges
        associations = dassoc.perform_data_association()
        for assoc in associations:
            # graphslam.view_solution()
            i = assoc[0]
            j = assoc[1]
            atb = keyframe_manager.compute_transformation_global(i, j)
            graphslam.add_non_consecutive_observation(i, j, atb)
            # keyframe_manager.view_map()
        if len(associations):
            # graphslam.view_solution()
            # optimizing whenever non_consecutive observations are performed (loop closing)
            graphslam.optimize()
            # graphslam.view_solution()
            keyframe_manager.save_solution(graphslam.get_solution())
            # keyframe_manager.view_map(xgt=odo_gt)

        # graphslam.view_solution()
        # or optimizing at every new observation
        # graphslam.optimize()

    graphslam.view_solution()
    keyframe_manager.save_solution(graphslam.get_solution())
    keyframe_manager.view_map(xgt=odo_gt, sampling=10)
    # or optimizing when all information is available
    graphslam.optimize()


if __name__ == "__main__":
    main()
