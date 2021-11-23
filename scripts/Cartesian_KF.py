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

from odom_set import odom_set

from sympy import *
first_call = True
x = 0.0
y = 0.0
theta = 0.0

# do I need to change these?
theta_0 = 0.0
x_0 = 0.0
y_0 = 0.0

dist_thresh = np.inf

# cartesisan state of system[5*np.pi/4]-target relatives is:
# [v_x, v_y, r_x, r_y]

# modified polar state of system-target relatives is:
# [beta_dot, r_dot / r, beta, 1 / r]
# initial guess
#y_kmin1_kmin1 = np.array([[0.0], [0.0], [2.0], [1.0]], dtype=float)
x_kmin1_kmin1 = np.array([[0.0], [0.0], [0.854], [-0.453]], dtype=float)
P_kmin1_kmin1 = np.array([[10, 0.0, 0.0, 0.0],
                          [0.0, 10, 0.0, 0.0],
                          [0.0, 0.0, 10, 0.0],
                          [0.0, 0.0, 0.0, 10]], dtype=float)

# t0 = (k - 1) * T, T is sampling rate
t0 = time.time()

# tracking own acceleration between t0 and t
a_ox = np.array([], dtype=float)
a_oy = np.array([], dtype=float)

# question to answer: is this just a single value, or are these updates all at the same rate?

H = np.array([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]], dtype=float)

R = 0.01
first = True


def odom_callback(odom_msg):
    global x, y, theta, first_call
    x, y, theta, first_call = odom_set(odom_msg, first_call, theta_0, x_0, y_0)


# we are assuming a constant velocity of the object
def imu_callback(imu_msg):
    global a_ox, a_oy
    a_ox = np.append(a_ox, imu_msg.linear_acceleration.x)
    a_oy = np.append(a_oy, imu_msg.linear_acceleration.y)

# computating the own-acceleration terms, these are treated as noise in this problem
def compute_accels(a_ox, a_oy, t0, t):
    x_tspan = np.linspace(t0, t, len(a_ox))
    y_tspan = np.linspace(t0, t, len(a_oy))

    #rospy.loginfo('x_tspan size: {0}'.format(np.shape(x_tspan)))
    #rospy.loginfo('y_tspan size: {0}'.format(np.shape(y_tspan)))

    #rospy.loginfo('a_ox size: {0}'.format(np.shape(a_ox)))
    #rospy.loginfo('a_oy size: {0}'.format(np.shape(a_oy)))

    w_o1 = np.trapz(a_ox, x_tspan)
    w_o2 = np.trapz(a_oy, y_tspan)
    w_o3 = np.trapz((t - x_tspan)*a_ox, x_tspan)
    w_o4 = np.trapz((t - y_tspan)*a_oy, y_tspan)
    #rospy.loginfo('w_o1: {0}'.format(w_o1))
    #rospy.loginfo('w_o2: {0}'.format(w_o2))

    #rospy.loginfo('w_o3: {0}'.format(w_o3))
    #rospy.loginfo('w_o4: {0}'.format(w_o4))
    w = -np.array([[w_o1], [w_o2], [w_o3], [w_o4]], dtype=float)

    return w

def kf_update_loop(y_t):
    global first, a_ox, a_oy, t0, x_kmin1_kmin1, P_kmin1_kmin1
    t = time.time()
    print('time period: ', t - t0)
    print('sensor reading: ', y_t)
    # calculate accelerative noise
    w = compute_accels(a_ox, a_oy, t0, t)
    print('w, accelerative noise: ', w)
    print('y_kmin1_kmin1, initial state: ', y_kmin1_kmin1)
    # Calculating y( K / K - 1), state estimate step
    # x_kmin1_kmin1 = fx_operator(y_kmin1_kmin1)
    print('x_kmin1_kmin1, state estimate in Cartesian: ', x_kmin1_kmin1)
    #rospy.loginfo('fx_y_t0: {0}'.format(fx_y_t0))
    A_x_t_t0 = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [t - t0, 0, 1, 0],
                [0, t - t0, 0, 1]]
    print('A_x_t_t0, transition matrix: ', A_x_t_t0)
    x_k_kmin1 = np.matmul(A_x_t_t0, x_kmin1_kmin1) + w
    print('x_k_kmin1, state estimate in Cartesian: ', x_k_kmin1)

    #rospy.loginfo('x_t: {0}'.format(x_t))
    y_k_kmin1 = fy_operator(x_k_kmin1)
    print('y_k_kmin1, state estimate in MP: ', y_k_kmin1)

    # Calculating A_y estimate, A matrix for MP coordinates
    A_y = compute_A_y(x_k_kmin1, y_kmin1_kmin1, t0, t, w)
    print('A_y, Jacobian of MP matrix: ', A_y)

    # Calculating covariance matrix estimate
    P_k_kmin1 = np.matmul(np.matmul(A_y, P_kmin1_kmin1), np.transpose(A_y, (0, 1)))
    print('P_k_kmin1, covariance estimate: ', P_k_kmin1)
    # Calculating Kalman gain
    H_tranpose = np.transpose(H)
    pt1 = np.matmul(P_k_kmin1, H_tranpose)
    pt2 = np.matmul(np.matmul(H, P_k_kmin1), H_tranpose)
    pt3 = np.random.normal(loc=0, scale=R, size=1) * np.eye(len(H))

    print('pt1: ', pt1)
    print('pt2: ', pt2)
    print('pt3: ', pt3)
    print('pt2 + pt3: ', pt2+pt3)
    # if H is only one value, inverse needs to be removed
    # NOTE REMOVING R FOR NOW, TRUSTING MEASUREMENTS
    if len(H) > 1:
        G_k = np.matmul(pt1, np.linalg.inv(pt2 + pt3))
    else:
        G_k = pt1 * (1 / (pt2 + pt3))
    print('G_k, Kalman gain: ', G_k)
    # Calculating the state update
    eta_t = np.random.normal(loc=0, scale=R, size=1)
    if len(H) > 1:
        beta_tilde_k = np.reshape(np.matmul(H, y_t) + eta_t, (2, 1))
    else:
        beta_tilde_k = np.matmul(H, y_t) + eta_t

    print('beta_tilde_k, sensor measurement: ', beta_tilde_k)
    y_k_k = y_k_kmin1 + np.matmul(G_k, (beta_tilde_k - np.matmul(H, y_k_kmin1)))
    print('y_k_k, state update: ', y_k_k)

    # Calculating the covariance update
    P_k_k = np.matmul((np.eye(4) - np.matmul(G_k, H)), P_k_kmin1)
    print('P_k_k: ', P_k_k)

    # setting old estimates equal to new estimates
    y_kmin1_kmin1 = y_k_k
    P_kmin1_kmin1 = P_k_k
    if first:
        plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(121)
    plt.title('Bearing comparison')
    plt.scatter(t, y_k_kmin1[2], c='r', marker='o', label='Bearing estimate')
    plt.scatter(t, y_t[2], c='k', marker='^', label='Bearing measurement')
    plt.ylim([-6.28, 6.28])
    plt.xlim([t - 5, t + 5])
    #@ax1.legend('Prediction', 'Sensor')
    if first:
        ax1.legend(loc="upper right")
    ax2 = plt.subplot(122)
    plt.title('Reciprocal range comparison')
    plt.scatter(t, y_k_kmin1[3], c='r', marker='o', label='Reciprocal range estimate')
    plt.scatter(t, y_t[3], c='k', marker='^', label='Reciprocal range measurement')
    plt.ylim([0, 4.0])
    plt.xlim([t - 5, t + 5])
    if first:
        ax2.legend(loc="upper right")
        first = False
    plt.draw()
    plt.pause(0.0000001)
    # resetting accelerations

    a_ox = np.array([], dtype=float)
    a_oy = np.array([], dtype=float)
    t0 = t

def scan_callback(scan_msg):
    global a_ox, a_oy, t0, P_kmin1_kmin1, y_kmin1_kmin1

    # cluster object
    objs_bearing_cluster = []
    objs_range_cluster = []
    obj_bearing_cluster = []
    obj_range_cluster = []
    for deg_i in range(1,len(scan_msg.ranges)):
        dist_i = scan_msg.ranges[deg_i]
        rad_i = deg_i * np.pi / 180
        dist_diff = np.abs(scan_msg.ranges[deg_i] - scan_msg.ranges[deg_i - 1])
        if dist_i < dist_thresh and dist_diff < 0.3:
            obj_range_cluster.append(dist_i)
            obj_bearing_cluster.append(rad_i)
        else:
            if len(obj_bearing_cluster) > 0:
                objs_bearing_cluster.append(obj_bearing_cluster)
                objs_range_cluster.append(obj_range_cluster)
                break
                #obj_bearing_cluster = []
                #obj_range_cluster = []
    rospy.loginfo('objs_bearing_cluster size: {0}'.format(np.shape(objs_bearing_cluster)))
    #print('obj_bearing_cluster: ', obj_bearing_cluster)
    #print('obj_range_cluster: ', obj_range_cluster)

    obj_range = np.mean(objs_range_cluster[0])
    obj_bearing = np.mean(objs_bearing_cluster[0])
    robot_to_world_T = [[np.cos(theta), -np.sin(theta), x],
                        [np.sin(theta), np.cos(theta), y],
                        [0, 0, 1]]

    lidar_pt_x = obj_range * np.cos(obj_bearing)  # IN LIDAR FRAME
    lidar_pt_y = obj_range * np.sin(obj_bearing)  # IN LIDAR FRAME

    lidar_pt = [[lidar_pt_x], [lidar_pt_y], [1]]
    world_pt = np.matmul(robot_to_world_T, lidar_pt)
    print('lidar_pt: ', lidar_pt)
    lidar_endpoint = np.array([world_pt[0][0], world_pt[1][0]])
    # robot state: x, y, theta
    # object state: lidar_endpoint[0], lidar_endpoint[1]
    print('lidar_endpoint: ', lidar_endpoint)
    print('x, y,', (x,y))
    r_vector = [lidar_endpoint[0] - x, lidar_endpoint[1] - y] # obstacle_state - robot_state
    # here, bearing ranges from -pi to pi

    y_t = np.array([0.0, 0.0, np.arctan2(r_vector[0], r_vector[1]), 1.0 / np.linalg.norm(r_vector)], dtype=float)
    kf_update_loop(y_t)
    #print('object point: ', lidar_endpoint)

## return relative position
## return relative velocity

def MP_KF():
    rospy.init_node('plot_lidar', anonymous=True)
    # plt.figure(figsize=(15, 5))
    odom_sub = rospy.Subscriber('/odom', Odometry, odom_callback, queue_size=3, buff_size=2**24)
    imu_sub = rospy.Subscriber('/imu', Imu, imu_callback, queue_size=3, buff_size=2**24)
    scan_sub = rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1, buff_size=2**24)
    # obtain_symbolic_derivatives()
    plt.ion()
    plt.show()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    MP_KF()


