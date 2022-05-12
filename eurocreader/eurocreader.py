import numpy as np
from tools.quaternion import Quaternion
import pandas as pd


class EurocReader():
    def __init__(self, directory):
        self.directory = directory

    def prepare_experimental_data(self, deltaxy, deltath, nmax_scans=None):
        print("PREPARING EXPERIMENT DATA")
        # eurocreader = EurocReader(directory=directory)
        # sample odometry at deltaxy and deltatheta
        odometry_times = self.sample_odometry(deltaxy=deltaxy, deltath=deltath)
        if nmax_scans is not None:
            print("CAUTION: CROPPING DATA TO: ", nmax_scans)
            odometry_times = odometry_times[0:nmax_scans]

        # read dfs from data
        df_odometry = self.read_odometry_data()
        df_scan_times = self.read_scan_times()
        df_ground_truth = self.read_ground_truth_data()
        # for every time in odometry_times, find the closest times of a scan.
        # next, for every time of the scan, find again the closest odometry and the closest ground truth
        scan_times, _, _ = self.get_closest_data(df_scan_times, odometry_times)
        _, odo_pos, odo_orient = self.get_closest_data(df_odometry, scan_times)
        _, gt_pos, gt_orient = self.get_closest_data(df_ground_truth, scan_times)
        print("FOUND: ", len(scan_times), "TOTAL SCANS")
        return scan_times, gt_pos, gt_orient

    def read_ground_truth_data(self):
        gt_csv_filename = self.directory + '/robot0/ground_truth/data.csv'
        df_gt = pd.read_csv(gt_csv_filename)
        return df_gt

    def read_odometry_data(self):
        odo_csv_filename = self.directory + '/robot0/odom/data.csv'
        df_odo = pd.read_csv(odo_csv_filename)
        return df_odo

    def read_scan_times(self):
        scan_times_csv_filename = self.directory + '/robot0/lidar/data.csv'
        df_scan_times = pd.read_csv(scan_times_csv_filename)
        return df_scan_times

    def sample_odometry(self, deltaxy=0.5, deltath=0.2):
        """
        Get odometry times separated by dxy (m) and dth (rad)
        """
        df_odo = self.read_odometry_data()
        odo_times = []
        for ind in df_odo.index:
            # print(df_odo['x'][ind])
            position = [df_odo['x'][ind], df_odo['y'][ind], df_odo['z'][ind]]
            q = Quaternion([df_odo['qw'][ind], df_odo['qx'][ind], df_odo['qy'][ind], df_odo['qz'][ind]])
            th = q.Euler()
            odo = np.array([position[0], position[1], th.abg[2]])
            current_time = df_odo['#timestamp [ns]'][ind]
            if ind == 0:
                odo_times.append(current_time)
                odoi = odo
            odoi1 = odo

            dxy = np.linalg.norm(odoi1[0:2]-odoi[0:2])
            dth = np.linalg.norm(odoi1[2]-odoi[2])
            if dxy > deltaxy or dth > deltath:
                odo_times.append(current_time)
                odoi = odoi1
        return np.array(odo_times)

    def get_closest_scan_times(self, odometry_times):
        df_scan_times = self.read_scan_times()
        scan_times = []
        # now find closest times to odo times
        for timestamp in odometry_times:
            result_index = df_scan_times['#timestamp [ns]'].sub(timestamp).abs().idxmin()
            scan_times.append(df_scan_times['#timestamp [ns]'][result_index])
        return scan_times

    def get_closest_odometry(self, odometry_times):
        df_odo = self.read_odometry_data()
        odometry = []
        # now find odo corresponding to closest times
        for timestamp in odometry_times:
            ind = df_odo['#timestamp [ns]'].sub(timestamp).abs().idxmin()
            position = [df_odo['x'][ind], df_odo['y'][ind], df_odo['z'][ind]]
            q = Quaternion([df_odo['qw'][ind], df_odo['qx'][ind], df_odo['qy'][ind], df_odo['qz'][ind]])
            th = q.Euler()
            odo = np.array([position[0], position[1], th.abg[2]])
            odometry.append(odo)
        return odometry

    def get_closest_data(self, df_data, time_list):
        # df_odo = self.read_odometry_data()
        positions = []
        orientations = []
        corresp_time_list = []
        # now find odo corresponding to closest times
        for timestamp in time_list:
            # find the closest timestamp in df
            ind = df_data['#timestamp [ns]'].sub(timestamp).abs().idxmin()
            try:
                position = [df_data['x'][ind], df_data['y'][ind], df_data['z'][ind]]
                positions.append(position)
            except:
                pass
            try:
                orientation = [df_data['qx'][ind], df_data['qy'][ind], df_data['qz'][ind], df_data['qw'][ind]]
                orientations.append(orientation)
            except:
                pass
            corresp_time = df_data['#timestamp [ns]'][ind]
            corresp_time_list.append(corresp_time)
        return np.array(corresp_time_list), np.array(positions), np.array(orientations)

