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

# modified polar state of system-target relatives is:
# [beta, 1/r, betadot, rdot / r]
# initial guess of state


# tracking own acceleration between t0 and t
a_ox = np.array([], dtype=float)
a_oy = np.array([], dtype=float)


class MPKalmanFilter:
    def __init__(self):
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=float)
        self.R = np.eye(2) * 0.0001  # measurement noise
        self.Q = np.eye(4) * 0.005  # covariance noise
        self.y = np.array([[2.0], [1.0], [0.0], [0.0]], dtype=float)  # MP state
        self.P = np.array([[10.0e-6, 0.0, 0.0, 0.0],
                           [0.0, 10.0e-4, 0.0, 0.0],
                           [0.0, 0.0, 10.0e-3, 0.0],
                           [0.0, 0.0, 0.0, 10e-4]], dtype=float)  # covariance matrix
        self.G = np.array([[1.0], [1.0], [1.0], [1.0]], dtype=float) # Kalman gain
        self.t0 = time.time()
        self.t = time.time()
        self.a = np.array([[0.0], [0.0]])  # horizontal and vertical acceleration of system
        self.x_obs = 0.0  # x pos of observer
        self.y_obs = 0.0  # y pos of observer
        self.theta_obs = 0.0  # theta of observer
        self.first_call = True  # first odometry call
        self.first_update_call = True # first KF update call (for plotting)
        self.y_tilde = np.array([[0.0], [0.0]])
        self.A = np.zeros((4, 4), dtype=float)
        # do I need to change these?
        self.theta_0 = 0.0
        self.x_0 = 0.0
        self.y_0 = 0.0

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
        print('updating state: x, y, theta', (self.x_obs, self.y_obs, self.theta_obs))
        print('x_0, y_0, theta_0', (self.x_0, self.y_0, self.theta_0))



    # callback for /imu topic, obtaining the acceleration of the system
    def imu_callback(self, imu_msg):
        self.a[0] = imu_msg.linear_acceleration.x
        self.a[1] = imu_msg.linear_acceleration.y

    # forward propagate state using euler integration and equation B3 from Aidala/Hammel
    def integrate(self):
        self.t = time.time()
        self.T = self.t - self.t0
        self.y[0] = self.y[0] + self.y[2]*self.T  # beta
        self.y[1] = self.y[1] + (-self.y[1]*self.y[3])*self.T  # 1 / r
        # NOTE: I CHANGED THE SIGN BEFORE THE LAST TERM. A/H HAVE BETA CW BUT NORMALLY IT IS CCW
        self.y[2] = self.y[2] + (-2 * self.y[2]*self.y[3] -
                                 self.y[1]*(self.a[0]*np.cos(self.y[0]) - self.a[1]*np.sin(self.y[0])))*self.T # betadot
        # NOTE: THIS HAS NOT BEEN VERIFIED YET
        self.y[3] = self.y[3] + (self.y[2]*self.y[2] - self.y[3]*self.y[3] -
                                 self.y[1]*(self.a[0]*np.sin(self.y[0]) + self.a[1]*np.cos(self.y[0])))*self.T # rdot / r

    #  linearize the nonlinear MP dynamics, using the discrete update equation
    def linearize(self):
        # A = np.zeros((4, 4), dtype=float)
        a_beta = self.a[0]*np.cos(self.y[0]) - self.a[1]*np.sin(self.y[0])
        a_r = self.a[0]*np.sin(self.y[0]) + self.a[1]*np.cos(self.y[0])
        # first row corresponds to beta
        # second row corresponds to 1/r
        # third row corresponds to betadot
        # fourth row corresponds to rdot / r
        self.A = np.array([[1.0, 0.0, self.T, 0.0],
                      [0.0, 1 - self.T, 0.0, self.T],
                      [0.0, -a_beta*self.T, 1.0 - 2.0*self.T, -2.0*self.T],
                      [0.0, -a_r*self.T, 2*self.y[2], 1.0 - 2.0*self.y[3]]], dtype=float)

    def scan_callback(self, scan_msg):
        # cluster object
        obj_bearing_cluster = []
        obj_range_cluster = []
        for deg_i in range(1, len(scan_msg.ranges)):
            dist_i = scan_msg.ranges[deg_i]
            rad_i = deg_i * np.pi / 180
            # dist_diff = np.abs(scan_msg.ranges[deg_i] - scan_msg.ranges[deg_i - 1])
            if dist_i < dist_thresh:
                obj_range_cluster.append(dist_i)
                obj_bearing_cluster.append(rad_i)

        obj_range = np.mean(obj_range_cluster)
        obj_bearing = np.mean(obj_bearing_cluster)
        # print('obj range: ', obj_range)
        # print('obj bearing: ', obj_bearing)
        robot_to_world_T = [[np.cos(self.theta_obs), -np.sin(self.theta_obs), self.x_obs],
                            [np.sin(self.theta_obs), np.cos(self.theta_obs), self.y_obs],
                            [0, 0, 1]]

        lidar_pt_x = obj_range * np.cos(obj_bearing)  # IN LIDAR FRAME
        lidar_pt_y = obj_range * np.sin(obj_bearing)  # IN LIDAR FRAME

        lidar_pt = [[lidar_pt_x], [lidar_pt_y], [1]]
        world_pt = np.matmul(robot_to_world_T, lidar_pt)
        #print('lidar_pt: ', lidar_pt)
        lidar_endpoint = np.array([world_pt[0][0], world_pt[1][0]])
        #lidar_endpoint = np.array([0.9, -0.6])
        # robot state: x, y, theta
        # object state: lidar_endpoint[0], lidar_endpoint[1]
        # print('lidar_endpoint: ', lidar_endpoint)
        r_vector = [lidar_endpoint[0] - self.x_obs, lidar_endpoint[1] - self.y_obs]  # obstacle_state - robot_state
        # here, bearing ranges from -pi to pi
        # print('r_vector: ', r_vector)
        self.y_tilde = np.array([[np.arctan2(r_vector[0], r_vector[1])],
                                 [1.0 / np.linalg.norm(r_vector)]], dtype=float)

    #def measure(self):
    #

    def kf_update_loop(self):
        self.integrate()
        self.linearize()
        ## DISCRETIZE NOISE? NOT REALLY SURE WHAT IS GOING ON HERE
        ## MEASURE? DOING THIS ON SCAN_CALLBACK
        ## UPDATE P
        ## NOTE: CHECK THIS TRANSPOSE
        A_t = np.transpose(self.A, (1, 0))
        print('A: ', self.A)
        print('A_t: ', A_t)
        self.P = np.matmul(np.matmul(self.A, self.P), A_t) + self.Q
        self.G = np.matmul(np.matmul(self.P, np.transpose(self.H, (1,0))),
                           np.linalg.inv(np.matmul(np.matmul(self.H, self.P), np.transpose(self.H, (1, 0))) + self.R))
        print('G size: ', np.shape(self.G))
        pt1 = np.matmul(self.H, self.y)
        print('pt1 size: ', np.shape(pt1))
        print('y tilde: ', self.y_tilde)
        pt2 = self.y_tilde - pt1
        print('pt2 size: ', np.shape(pt2))
        self.y = self.y + np.matmul(self.G, pt2)
        self.P = np.matmul((np.eye(4) - np.matmul(self.G, self.H)), self.P)
        print('y after: ', self.y)
        print('P after: ', self.P)
        ## CORRECT

        if self.first_update_call:
            plt.figure(figsize=(15, 4))
        ax1 = plt.subplot(121)
        plt.title('Bearing comparison')
        plt.scatter(self.t, self.y[0], c='r', marker='o', label='Bearing estimate')
        plt.scatter(self.t, self.y_tilde[0], c='k', marker='^', label='Bearing measurement')
        plt.ylim([-6.28, 6.28])
        plt.xlim([self.t - 5, self.t + 5])
        #@ax1.legend('Prediction', 'Sensor')
        if self.first_update_call:
            ax1.legend(loc="upper right")
        ax2 = plt.subplot(122)
        plt.title('Reciprocal range comparison')
        plt.scatter(self.t, self.y[1], c='r', marker='o', label='Reciprocal range estimate')
        plt.scatter(self.t, self.y_tilde[1], c='k', marker='^', label='Reciprocal range measurement')
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


