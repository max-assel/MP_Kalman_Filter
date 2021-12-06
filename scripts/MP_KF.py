#!/usr/bin/env python
#import matplotlib
#matplotlib.use('Agg')
import rospy
import numpy as np
import time
import copy

import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist

dist_thresh = np.inf
np.set_printoptions(linewidth=200)
# modified polar state of system-target relatives is:
# [beta, 1/r, betadot, rdot / r]
# initial guess of state


# tracking own acceleration between t0 and t
a_ox = np.array([], dtype=float)
a_oy = np.array([], dtype=float)


##################################################
# Python code that implements the modified polar #
# coordinates-based extended Kalman filter. One  #
# Slight change to coordinates: sin(beta) and    #
# cos(beta) used instead of beta for stability   #
##################################################

class MPKalmanFilter:
    def __init__(self):
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float)
        self.R = np.eye(3) * 0.00001  # measurement noise
        self.Q = np.eye(5) * 0.005  # covariance noise
        ## CHOOSING A STATE OF 1/R, BETA, RDOT/R, BETADOT
        self.y_left = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]], dtype=float)  # MP state
        self.y_right = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]], dtype=float)  # MP state

        self.P_left = np.array([[10.0e-6, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 10.0e-4, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 10.0e-4, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 10e-4, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 10e-3]], dtype=float)  # covariance matrix
        self.P_right = np.array([[10.0e-6, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 10.0e-4, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 10.0e-4, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 10e-4, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 10e-3]], dtype=float)  # covariance matrix
        self.G_left = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype=float)  # Kalman gain
        self.G_right = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype=float)  # Kalman gain
        self.t0 = time.time()
        self.t = time.time()
        self.T = self.t - self.t0
        self.a = np.array([[0.0], [0.0]])  # horizontal and vertical acceleration of system
        self.x_obs = 0.0  # x pos of observer
        self.y_obs = 0.0  # y pos of observer
        self.theta_obs = 0.0  # theta of observer
        self.robot_to_world_T = [[np.cos(self.theta_obs), -np.sin(self.theta_obs), self.x_obs],
                            [np.sin(self.theta_obs), np.cos(self.theta_obs), self.y_obs],
                            [0, 0, 1]]

        self.first_call = True  # first odometry call
        self.first_update_call = True  # first KF update call (for plotting)
        self.y_tilde_left = np.array([[0.0], [0.0], [0.0]])
        self.y_tilde_right = np.array([[0.0], [0.0], [0.0]])

        self.A_left = np.zeros((5, 5), dtype=float)
        self.A_right = np.zeros((5, 5), dtype=float)

        # do I need to change these?
        self.theta_0 = 0.0
        self.x_0 = 0.0
        self.y_0 = 0.0
        self.global_goal = np.array([5.0, 0.0], dtype=float)
        self.r_ins = 0.1


    # SETS ORIGINAL POSITION TO GLOBL ORIGIN
    def odom_callback(self, odom_msg):
        position = odom_msg.pose.pose.position

        # Orientation uses the quaternion parameterization.
        # To get the angular position along the z-axis, the following equation is required.
        q = odom_msg.pose.pose.orientation
        # returns value between -pi and pi
        orientation = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        if self.first_call:
            # The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.first_call = False
            self.theta_0 = orientation
            # theta = theta_0
            Mrot = np.matrix([[np.cos(self.theta_0), np.sin(self.theta_0)], [-np.sin(self.theta_0), np.cos(self.theta_0)]])
            self.x_0 = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y
            self.y_0 = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y

        Mrot = np.matrix([[np.cos(self.theta_0), np.sin(self.theta_0)], [-np.sin(self.theta_0), np.cos(self.theta_0)]])

        # We subtract the initial values
        self.x_obs = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y - self.x_0
        self.y_obs = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y - self.y_0
        self.theta_obs = orientation - self.theta_0
        self.robot_to_world_T = [[np.cos(self.theta_obs), -np.sin(self.theta_obs), self.x_obs],
                            [np.sin(self.theta_obs), np.cos(self.theta_obs), self.y_obs],
                            [0, 0, 1]]


    # callback for /imu topic, obtaining the acceleration of the system
    def imu_callback(self, imu_msg):
        self.a[0] = imu_msg.linear_acceleration.x
        self.a[1] = imu_msg.linear_acceleration.y

    # forward propagate state using euler integration and equation B3 from Aidala/Hammel
    def integrate(self, y):
        self.t = time.time()
        self.T = self.t - self.t0
        # entries are 1/r, beta, rdot/r, betadot
        new_y = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=float)  # MP state

        # 1 / r
        new_y[0] = y[0] + (-y[3]*y[0])*self.T
        # sin(beta)
        new_y[1] = y[1] + y[2]*y[4]*self.T
        # cos(beta)
        new_y[2] = y[2] + (-y[1])*y[4]*self.T
        # rdot / r,  NOTE: THIS HAS NOT BEEN VERIFIED YET
        new_y[3] = y[3] + (y[4]*y[4] - y[3]*y[3] - y[0] * (self.a[0]*y[1] + self.a[1]*y[2])) * self.T

        #  # betadot, NOTE: I CHANGED THE SIGN BEFORE THE LAST TERM. A/H HAVE BETA CW BUT NORMALLY IT IS CCW
        new_y[4] = y[4] + (-2 * y[3]*y[4] - y[0]*(self.a[0]*y[2] - self.a[1]*y[1]))*self.T
        return new_y

    #  linearize the nonlinear MP dynamics, using the discrete update equation
    def linearize(self, y):
        # A = np.zeros((4, 4), dtype=float)
        a_r = self.a[0]*y[1] + self.a[1]*y[2]
        a_beta = self.a[0]*y[2] - self.a[1]*y[1]

        #A = np.array([[1.0 - y[2]*self.T, 0.0, -y[0]*self.T, 0.0],
        #              [0.0, 1, 0.0, self.T],
        #              [-a_r*self.T, -y[0]*a_beta*self.T, 1.0 - 2.0*y[2]*self.T, 2.0*y[3]*self.T],
        #              [-a_beta*self.T, -y[0]*-a_r*self.T, -2.0*y[3]*self.T, 1.0 - 2.0*y[2]*self.T]], dtype=float)
        A = np.array([[1 - y[3]*self.T, 0, 0, -y[0]*self.T, 0],
                      [0, 1, y[4]*self.T, 0, y[2]*self.T],
                      [0, -y[4]*self.T, 1, 0, -y[1]*self.T],
                      [-a_r*self.T, -y[0]*self.a[0]*self.T, -y[0]*self.a[1]*self.T, 1 - 2*y[3]*self.T, 2*y[4]*self.T],
                      [-a_beta*self.T, y[0]*self.a[1]*self.T, -y[0]*self.a[0]*self.T, -2*y[4]*self.T, 1 - 2*y[3]*self.T]], dtype=float)
        return A

    def scan_callback(self, scan_msg):
        # cluster object
        #obj_bearing_cluster = []
        #obj_range_cluster = []
        self.gaps = []
        # print('ranges: ', scan_msg.ranges)
        for deg_i in range(1, len(scan_msg.ranges) - 1):
            dist_i = scan_msg.ranges[deg_i]
            rad_i = deg_i * np.pi / 180.0
            dist_iplus1 = scan_msg.ranges[deg_i + 1]
            rad_iplus1 = (deg_i + 1) * np.pi / 180.0
            # if current distance is finite and next distance is infinite, start a gap
            if dist_i < dist_thresh <= dist_iplus1:
                deg_j = deg_i + 1
                dist_j = dist_iplus1
                while dist_j >= dist_thresh:
                    dist_j = scan_msg.ranges[deg_j]
                    deg_j = (deg_j + 1) % 360
                rad_j = deg_j * np.pi / 180.0
                lidar_pt_i = [[dist_i*np.cos(rad_i)], [dist_i*np.sin(rad_i)], [1]]
                world_pt_i = np.matmul(self.robot_to_world_T, lidar_pt_i)
                lidar_pt_j = [[dist_j * np.cos(rad_j)], [dist_j * np.sin(rad_j)], [1]]
                world_pt_j = np.matmul(self.robot_to_world_T, lidar_pt_j)
                i_j_distance = np.linalg.norm(world_pt_i - world_pt_j)
                # see how long gap is
                if i_j_distance > 2*self.r_ins:
                    self.gaps.append([deg_i, deg_j])
                deg_i = deg_j - 1  # skip to end of gap in pass through laser scan
        # print('current gaps: ', self.gaps)
        current_gap_dist = np.inf
        current_gap = None
        for gap in self.gaps:
            # print('gap: ', gap)
            gap_right_deg = gap[0]
            gap_left_deg = gap[1]
            gap_left_range = scan_msg.ranges[gap_left_deg]
            gap_right_range = scan_msg.ranges[gap_right_deg]
            gap_right_rad = gap_right_deg * np.pi / 180.0
            gap_left_rad = gap_left_deg * np.pi / 180.0
            gap_left_lidar_pt = [[gap_left_range * np.cos(gap_left_rad)], [gap_left_range * np.sin(gap_left_rad)], [1]]
            gap_right_lidar_pt = [[gap_right_range * np.cos(gap_right_rad)], [gap_right_range * np.sin(gap_right_rad)], [1]]
            gap_left_world_pt = np.matmul(self.robot_to_world_T, gap_left_lidar_pt)
            gap_right_world_pt = np.matmul(self.robot_to_world_T, gap_right_lidar_pt)
            gap_mid_world_pt = (gap_left_world_pt + gap_right_world_pt) / 2.0
            gap_mid_xy = np.array([gap_mid_world_pt[0], gap_mid_world_pt[1]])
            gap_dist = np.linalg.norm(gap_mid_xy - self.global_goal)
            if gap_dist < current_gap_dist:
                current_gap = gap
                current_gap_dist = gap_dist
        # print('chosen gap: ', current_gap)
        gap_right_deg = current_gap[0]
        gap_left_deg = current_gap[1]
        gap_right_bearing = gap_right_deg * np.pi / 180.0
        gap_left_bearing = gap_left_deg * np.pi / 180.0
        gap_right_range = scan_msg.ranges[gap_right_deg]
        gap_left_range = scan_msg.ranges[gap_left_deg]

        # print('obj range: ', obj_range)
        # print('obj bearing: ', obj_bearing)

        left_pt_lidar = [[gap_left_range * np.cos(gap_left_bearing)], [gap_left_range * np.sin(gap_left_bearing)], [1]]
        right_pt_lidar = [[gap_right_range * np.cos(gap_right_bearing)], [gap_right_range * np.sin(gap_right_bearing)], [1]]

        left_pt_world = np.matmul(self.robot_to_world_T, left_pt_lidar)
        right_pt_world = np.matmul(self.robot_to_world_T, right_pt_lidar)

        # print('world pt: ', world_pt)
        #print('lidar_pt: ', lidar_pt)
        left_gap_point = np.array([left_pt_world[0][0], left_pt_world[1][0]])
        right_gap_point = np.array([right_pt_world[0][0], right_pt_world[1][0]])
        # lidar_endpoint = np.array([0.9, 0.5])
        # robot state: x, y, theta
        # object state: lidar_endpoint[0], lidar_endpoint[1]
        # print('lidar_endpoint: ', lidar_endpoint)
        range_left = [left_gap_point[0] - self.x_obs, left_gap_point[1] - self.y_obs]  # obstacle_state - robot_state
        range_right = [right_gap_point[0] - self.x_obs, right_gap_point[1] - self.y_obs]  # obstacle_state - robot_state
        # here, bearing ranges from -pi to pi
        # print('r_vector: ', r_vector)
        beta_tilde_left = np.arctan2(range_left[0], range_left[1])
        beta_tilde_right = np.arctan2(range_right[0], range_right[1])
        self.y_tilde_left = np.array([[1.0 / np.linalg.norm(range_left)],
                                      [np.sin(beta_tilde_left)],
                                      [np.cos(beta_tilde_left)]], dtype=float)
        self.y_tilde_right = np.array([[1.0 / np.linalg.norm(range_right)],
                                       [np.sin(beta_tilde_right)],
                                       [np.cos(beta_tilde_right)]], dtype=float)
    #def measure(self):
    #

    def kf_update_loop(self):
        self.y_left = self.integrate(self.y_left)
        self.y_right = self.integrate(self.y_right)
        self.A_left = self.linearize(self.y_left)
        self.A_right = self.linearize(self.y_right)
        ## DISCRETIZE NOISE? NOT REALLY SURE WHAT IS GOING ON HERE
        ## MEASURE? DOING THIS ON SCAN_CALLBACK
        ## UPDATE P
        ## NOTE: CHECK THIS TRANSPOSE
        A_left_t = np.transpose(self.A_left, (1, 0))
        A_right_t = np.transpose(self.A_right, (1, 0))
        # print('A: ', self.A)
        # print('A_t: ', A_t)
        self.P_left = np.matmul(np.matmul(self.A_left, self.P_left), A_left_t) + self.Q
        # print('P left: ', self.P_left)
        self.P_right = np.matmul(np.matmul(self.A_right, self.P_right), A_right_t) + self.Q
        # pt1 =
        # pt2 =
        # print('pt1: ', pt1)
        # print('pt2: ', pt2)
        self.G_left = np.matmul(np.matmul(self.P_left, np.transpose(self.H, (1, 0))),
                           np.linalg.inv(np.matmul(np.matmul(self.H, self.P_left), np.transpose(self.H, (1, 0))) + self.R))
        self.G_right = np.matmul(np.matmul(self.P_right, np.transpose(self.H, (1, 0))),
                                np.linalg.inv(np.matmul(np.matmul(self.H, self.P_right), np.transpose(self.H, (1, 0))) + self.R))

        # print('G size: ', np.shape(self.G))
        # pt1 =
        # print('pt1 size: ', np.shape(pt1))
        # print('y tilde: ', self.y_tilde)
        # pt2 =
        # print('pt2 size: ', np.shape(pt2))
        self.y_left = self.y_left + np.matmul(self.G_left, self.y_tilde_left - np.matmul(self.H, self.y_left))
        self.P_left = np.matmul((np.eye(5) - np.matmul(self.G_left, self.H)), self.P_left)
        self.y_right = self.y_right + np.matmul(self.G_right, self.y_tilde_right - np.matmul(self.H, self.y_right))
        self.P_right = np.matmul((np.eye(5) - np.matmul(self.G_right, self.H)), self.P_right)
        '''
        if np.linalg.norm(self.P_left) > 1000:
            self.P_left = np.array([[10.0e-6, 0.0, 0.0, 0.0], [0.0, 10.0e-4, 0.0, 0.0],
                                    [0.0, 0.0, 10.0e-3, 0.0], [0.0, 0.0, 0.0, 10e-4]], dtype=float)  # covariance matrix
            self.y_left = np.array([[1.0], [1.0], [0.0], [0.0]], dtype=float)  # MP state

        if np.linalg.norm(self.P_right) > 1000:
            self.P_right = np.array([[10.0e-6, 0.0, 0.0, 0.0], [0.0, 10.0e-4, 0.0, 0.0],
                                    [0.0, 0.0, 10.0e-3, 0.0], [0.0, 0.0, 0.0, 10e-4]], dtype=float)  # covariance matrix
            self.y_right = np.array([[1.0], [1.0], [0.0], [0.0]], dtype=float)  # MP state
        '''
        #print('y tilde left: ', self.y_tilde_left)
        #print('y left: ', self.y_left)
        #print('y_right: ', self.y_right)
        #print('y tilde right: ', self.y_tilde_right)
        # print('y after: ', self.y)
        # print('P after: ', self.P)
        ## CORRECT

        if self.first_update_call:
            plt.figure(figsize=(15, 4))
        ax1 = plt.subplot(121)
        plt.title('Bearing comparison')
        recovered_beta_tilde_left = np.arctan2(self.y_tilde_left[1], self.y_tilde_left[2])
        recovered_beta_tilde_right = np.arctan2(self.y_tilde_right[1], self.y_tilde_right[2])
        recovered_beta_left = np.arctan2(self.y_left[1], self.y_left[2])
        recovered_beta_right = np.arctan2(self.y_right[1], self.y_right[2])
        plt.scatter(self.t, recovered_beta_left, c='r', marker='o', label='Bearing estimate, left')
        plt.scatter(self.t, recovered_beta_tilde_left, c='r', marker='^', label='Bearing measurement, left')
        plt.scatter(self.t, recovered_beta_right, c='b', marker='o', label='Bearing estimate, right')
        plt.scatter(self.t, recovered_beta_tilde_right, c='b', marker='^', label='Bearing measurement, right')
        plt.ylim([-6.28, 6.28])
        plt.xlim([self.t - 5, self.t + 5])
        #@ax1.legend('Prediction', 'Sensor')
        if self.first_update_call:
            ax1.legend(loc="upper right")
        ax2 = plt.subplot(122)
        plt.title('Reciprocal range comparison')
        plt.scatter(self.t, self.y_left[0], c='g', marker='o', label='Reciprocal range estimate, left')
        plt.scatter(self.t, self.y_tilde_left[0], c='g', marker='^', label='Reciprocal range measurement, left')
        plt.scatter(self.t, self.y_right[0], c='k', marker='o', label='Reciprocal range estimate, right')
        plt.scatter(self.t, self.y_tilde_right[0], c='k', marker='^', label='Reciprocal range measurement, right')
        plt.ylim([0, 4.0])
        plt.xlim([self.t - 5, self.t + 5])
        if self.first_update_call:
            ax2.legend(loc="upper right")
            self.first_update_call = False
        plt.draw()
        plt.pause(0.0000001)
        # resetting accelerations

        #a_ox = np.array([], dtype=float)
        #a_oy = np.array([], dtype=float)
        #t0 = t
        self.t0 = self.t
    # kf_update_loop(y_t)
    #print('object point: ', lidar_endpoint)

## return relative position
## return relative velocity

def MP_KF():
    rospy.init_node('plot_lidar', anonymous=True)
    filter = MPKalmanFilter()
    odom_sub = rospy.Subscriber('/odom', Odometry, filter.odom_callback, queue_size=3, buff_size=2**24)
    imu_sub = rospy.Subscriber('/imu', Imu, filter.imu_callback, queue_size=3, buff_size=2**24)
    scan_sub = rospy.Subscriber('/scan', LaserScan, filter.scan_callback, queue_size=1, buff_size=2**24)
    plt.ion()
    plt.show()
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        filter.kf_update_loop()
        rate.sleep()

if __name__ == '__main__':
    MP_KF()


