#!/usr/bin/env python
#import matplotlib
#matplotlib.use('Agg')
import rospy
import numpy as np
import copy

import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist

from sklearn.metrics.pairwise import rbf_kernel
import time

dist_thresh = np.inf
first_call = True
theta_0 = 0
x_0 = 0
y_0 = 0
x = 0
y = 0
theta=0
endpt_x_s = []
endpt_y_s = []
free_x_s = []
free_y_s = []
ranges = []
eta = 0.3
epsilon = None
mu = None
sig = None

laser_point = Point()
#fig = plt.gcf()
#fig.show()
#fig.canvas.draw()

first = True

vel_dir_switch = 0
check_time = 0

tb3_1_vel_pub = rospy.Publisher('tb3_1/cmd_vel', Twist, queue_size=5)

def odom_callback(odom_msg):
    global x
    global y
    global theta
    global first_call, theta_0, x_0, y_0, theta, x, y
    global check_time, vel_dir_switch
    # 0 - 2: pos
    # 2 -4: zero
    # 4 - 6: neg
    # 6 - 8: zero
    cur_time = time.time()
    time_1 = 2
    time_2 = 7
    time_3 = 9
    time_4 = 14
    if cur_time - check_time < time_1:
        vel_dir_switch = 0
    elif cur_time - check_time > time_1 and cur_time - check_time < time_2:
        vel_dir_switch = 1
    elif cur_time - check_time > time_2 and cur_time - check_time < time_3:
        vel_dir_switch = 2
    elif cur_time - check_time > time_3 and cur_time - check_time < time_4:
        vel_dir_switch = 1
    else:
        check_time = cur_time

    twist = Twist()
    if vel_dir_switch == 0:
        twist.linear.x = 0.4
    elif vel_dir_switch == 1:
        twist.linear.x = 0
    elif vel_dir_switch == 2:
        twist.linear.x = -0.4

    tb3_1_vel_pub.publish(twist)

    position = odom_msg.pose.pose.position

    # Orientation uses the quaternion parameterization.
    # To get the angular position along the z-axis, the following equation is required.
    q = odom_msg.pose.pose.orientation
    # returns value between -pi and pi
    orientation = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    if first_call:
        # The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
        first_call = False
        theta_0 = orientation
        theta = theta_0
        Mrot = np.matrix([[np.cos(theta_0), np.sin(theta_0)], [-np.sin(theta_0), np.cos(theta_0)]])
        x_0 = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y
        y_0 = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y

    Mrot = np.matrix([[np.cos(theta_0), np.sin(theta_0)], [-np.sin(theta_0), np.cos(theta_0)]])

    # We subtract the initial values
    x = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y - x_0
    y = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y - y_0
    theta = orientation - theta_0

def gen_lidar_data(scan_msg):
    global x, y, endpt_x_s, endpt_y_s, free_x_s, free_y_s, theta, ranges, laser_point
    start_time = time.time()
    current_theta = theta
    current_x = x
    current_y = y
    new_endpt_x_s = []
    new_endpt_y_s = []
    new_free_x_s = []
    new_free_y_s = []
    points = []
    labels = []

    unoccupied_points_per_meter = 1.25
    margin = 0.01
    n = 0

    for deg in range(0,len(scan_msg.ranges)):
        dist = scan_msg.ranges[deg]
        if dist < dist_thresh:
            robot_to_world_T = [[np.cos(current_theta),  -np.sin(current_theta), current_x],
                                [np.sin(current_theta), np.cos(current_theta), current_y],
                                [0, 0, 1]]
            lidar_rad = deg*np.pi / 180

            para = np.sort(np.random.random(np.int16(dist * unoccupied_points_per_meter)) * (1 - 2 * margin) + margin)[
                   :, np.newaxis]

            lidar_pt_x = dist*np.cos(lidar_rad)  # IN LIDAR FRAME
            lidar_pt_y = dist*np.sin(lidar_rad)  # IN LIDAR FRAME

            lidar_pt = [[lidar_pt_x],[lidar_pt_y], [1]]
            world_pt = np.matmul(robot_to_world_T, lidar_pt)

            robot_pos = np.array((x, y))
            lidar_endpoint = np.array([world_pt[0][0], world_pt[1][0]])

            points_scan_i = robot_pos + para * (lidar_endpoint - robot_pos)

            new_endpt_x_s = np.append(new_endpt_x_s, world_pt[0], axis=0)
            new_endpt_y_s = np.append(new_endpt_y_s, world_pt[1], axis=0)
            new_free_x_s = np.append(new_free_x_s, points_scan_i[:, 0], axis=0)
            new_free_y_s = np.append(new_free_y_s, points_scan_i[:, 1], axis=0)
            # rospy.loginfo('deg: {0}'.format(deg))
            if n == 0:  # first data point
                points = np.vstack((points_scan_i, lidar_endpoint))
                labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))
                n += 1
            else:
                points = np.vstack((points, np.vstack((points_scan_i, lidar_endpoint))))
                labels = np.vstack((labels, np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))))
                '''
                if not first:
                    qPhi = rbf_kernel(points, grid, gamma=gamma)
                    # qw = np.random.multivariate_normal(mu, np.diag(sig), 100)
                    occ = sigmoid(qw.dot(qPhi.T))
                    trainingMean = np.mean(occ, axis=0)
                    #print('trainingMean shape: ', np.shape(trainingMean))
                    #print('trainingMean: ', trainingMean)
                    #print('points shape: ', np.shape(points))
                    #print('labels shape: ', np.shape(labels))
                    #print('labels: ', labels)

                    label_diff = np.squeeze(labels) - trainingMean
                    # print('label diff: ', label_diff)
                    mask = np.abs(label_diff) > eta
                    # print('mask: ', mask)

                    points = points[mask]
                    labels = labels[mask]
                '''

    #plot_lidar_data(new_endpt_x_s, new_endpt_y_s, new_free_x_s, new_free_y_s)
    trainingData = np.hstack((points, labels))
    print('time to process lidar: ', time.time() - start_time)
    # rospy.loginfo('training data size: {0}'.format(np.shape(trainingData)))
    generate_HM(trainingData)


def calcPosterior(Phi, y, xi, mu0, sig0):
    logit_inv = sigmoid(xi)
    lam = 0.5 / xi * (logit_inv - 0.5)

    sig = 1. /(1./sig0 + 2*np.sum( (Phi.T**2)*lam, axis=1)) # note the numerical trick for the dot product

    mu = sig*(mu0/sig0 + np.dot(Phi.T, y - 0.5).ravel())

    return mu, sig

def sigmoid(x):
    return 1. / (1 + np.exp(-x))



xlims = [-5, 8] #[-8, 8]
ylims = [-5, 1] #[-15, 1]
xx, yy = np.meshgrid(np.linspace(xlims[0], xlims[1], 10), np.linspace(ylims[0], ylims[1], 10))
grid = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)))
gamma = 2

qxx, qyy = np.meshgrid(np.linspace(xlims[0], xlims[1], 90), np.linspace(ylims[0], ylims[1], 90))
qX = np.hstack((qxx.ravel().reshape(-1, 1), qyy.ravel().reshape(-1, 1)))

def generate_HM(trainingData):
    global first, mu, sig, qw
    start_time = time.time()
    # x = -2 y = -0.5
    # Step 0 - training data

    X3, y3 = trainingData[:, :2], trainingData[:, 2]
    # Step 1 - define hinge points

    # Step 2 - compute features
    Phi = rbf_kernel(X3, grid, gamma=gamma)
    # print('time for computing feature: ', time.time() - start_time)
    # Step 3 - estimate the parameters
    # Let's define the prior
    N, D = Phi.shape[0], Phi.shape[1]
    epsilon = np.ones(N)
    # print('time for initializing epsilon: ', time.time() - start_time)

    if first:
        mu = np.zeros(D)
        sig = 10000 * np.ones(D)
    #print('hinge points shape: ', np.shape(grid))
    #print('phi shape: ', Phi.shape)
    #print('epsilon shape: ', np.shape(epsilon))
    #print('mu shape: ', np.shape(mu))
    #print('sig shape: ', np.shape(sig))
    for i in range(2):
        # E-step
        mu, sig = calcPosterior(Phi, y3, epsilon, mu, sig)

        # M-step
        epsilon = np.sqrt(np.sum((Phi ** 2) * sig, axis=1) + (Phi.dot(mu.reshape(-1, 1)) ** 2).ravel())
    # print('time for EM: ', time.time() - start_time)

    # Step 4 - predict

    qPhi = rbf_kernel(qX, grid, gamma=gamma)
    qw = np.random.multivariate_normal(mu, np.diag(sig), 100)
    occ = sigmoid(qw.dot(qPhi.T))
    occMean = np.mean(occ, axis=0)
    occStdev = np.std(occ, axis=0)
    # print('occMean shape: ', np.shape(occMean))
    # print('occStdDev shape: ', np.shape(occStdev))

    print('time taken for prediction: ', time.time()-start_time)

    # Plot
    if first:
        plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(131)
    ax1.clear()
    plt.scatter(grid[:,0], grid[:,1], c='k', marker='o')
    plt.scatter(X3[:, 0], X3[:, 1], c=y3, marker='x', cmap='jet')
    if first:
        plt.colorbar()
    plt.title('Hinge points and dataset')
    plt.scatter(x, y, c='g', s=100, marker='*')

    plt.xlim(xlims)
    plt.ylim(ylims)
    ax2 = plt.subplot(132)
    ax2.clear()
    plt.scatter(qX[:, 0], qX[:, 1], c=occMean, s=4, cmap='jet', vmin=0, vmax=1)
    if first:
        plt.colorbar()
    plt.title('Occupancy probability - mean')
    plt.xlim(xlims)
    plt.ylim(ylims)
    ax3 = plt.subplot(133)
    ax3.clear()
    plt.scatter(qX[:, 0], qX[:, 1], c=occStdev, s=4, cmap='jet', vmin=0)
    if first:
        plt.colorbar()
        first = False
    plt.title('Occupancy probability - stdev')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.draw()
    plt.pause(0.0000001)


def plot_lidar_data(new_endpt_x_s, new_endpt_y_s, new_freept_x_s, new_freept_y_s):
    #rospy.loginfo('new_endpt_x_s: {0}'.format(new_endpt_x_s))
    #rospy.loginfo('new_endpt_y_s: {0}'.format(new_endpt_y_s))
    #rospy.loginfo('new_freept_x_s: {0}'.format(new_freept_x_s))
    #rospy.loginfo('new_freept_y_s: {0}'.format(new_freept_y_s))
    ax1 = plt.subplot(111)
    ax1.clear()
    plt.plot(new_endpt_x_s, new_endpt_y_s, 'ro', markersize=1)
    plt.plot(new_freept_x_s, new_freept_y_s, 'bo', markersize=1)
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.draw()
    plt.pause(0.0000001)


def plot_lidar():
    rospy.init_node('plot_lidar', anonymous=True)
    # plt.figure(figsize=(15, 5))
    odom_sub = rospy.Subscriber('/tb3_0/odom', Odometry, odom_callback, queue_size=3, buff_size=2**24)
    scan_sub = rospy.Subscriber('/tb3_0/scan', LaserScan, gen_lidar_data, queue_size=1, buff_size=2**24)
    plt.ion()
    plt.show()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    plot_lidar()