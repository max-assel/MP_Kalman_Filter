#!/usr/bin/env python
#import matplotlib
#matplotlib.use('Agg')
import rospy
import numpy as np
import time

import matplotlib.pyplot as plt
from sensor_msgs.msg import Imu

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
        self.y = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]], dtype=float)  # MP state

        self.P = np.array([[10.0e-6, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 10.0e-4, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 10.0e-4, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 10e-4, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 10e-3]], dtype=float)  # covariance matrix

        self.G = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype=float)  # Kalman gain

        self.t0 = time.time()
        self.t = time.time()
        self.T = self.t - self.t0
        self.a = np.array([[0.0], [0.0]])  # horizontal and vertical acceleration of system
        self.first_update_call = True  # first KF update call (for plotting)

        self.y_tilde = np.array([[0.0], [0.0], [0.0]])
        self.A = np.zeros((5, 5), dtype=float)
        self.Ad = np.zeros((5, 5), dtype=float)
        self.dQ = np.zeros((5, 5), dtype=float)

    # callback for /imu topic, obtaining the acceleration of the system
    def imu_callback(self, imu_msg):
        self.a[0] = imu_msg.linear_acceleration.x
        self.a[1] = imu_msg.linear_acceleration.y

    def f_x(self, y):
        x = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=float)
        x[0] = (1/y[0])*y[1]  # r_x
        x[1] = (1/y[0])*y[2]  # r_y
        x[2] = (1/y[0])*(y[3]*y[1] + y[4]*y[2])  # v_x
        x[3] = (1/y[0])*(y[3]*y[2] - y[4]*y[1])  # v_y
        return x

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
        new_y[4] = y[4] + (-2*y[3]*y[4] - y[0]*(self.a[0]*y[2] - self.a[1]*y[1]))*self.T
        return new_y

    #  linearize the nonlinear MP dynamics, using the discrete update equation
    def linearize(self, y):
        a_r = self.a[0]*y[1] + self.a[1]*y[2]
        a_beta = self.a[0]*y[2] - self.a[1]*y[1]


        self.A_cont = np.array([[-y[3], 0.0, 0.0, -y[0], 0.0],
                                [0.0, 0.0, y[4], 0.0, y[2]],
                                [0.0, -y[4], 0.0, 0.0, -y[1]],
                                [-a_r, -y[0]*self.a[0], -y[0]*self.a[1], -2*y[3], 2*y[4]],
                                [-a_beta, y[0]*self.a[1], -y[0]*self.a[0], -2*y[4], -2*y[3]]], dtype=float)

        self.A = np.array([[1 - y[3]*self.T, 0, 0, -y[0]*self.T, 0],
                           [0, 1, y[4]*self.T, 0, y[2]*self.T],
                           [0, -y[4]*self.T, 1, 0, -y[1]*self.T],
                           [-a_r*self.T, -y[0]*self.a[0]*self.T, -y[0]*self.a[1]*self.T, 1 - 2*y[3]*self.T, 2*y[4]*self.T],
                           [-a_beta*self.T, y[0]*self.a[1]*self.T, -y[0]*self.a[0]*self.T, -2*y[4]*self.T, 1 - 2*y[3]*self.T]], dtype=float)
        # return A  #

    def discretizeQ(self):
        self.dQ = self.Q * self.T

        M2 = 0.5 * self.T * (np.transpose(self.A_cont * self.dQ, (1, 0)) + self.A_cont*self.dQ)
        M3 = (1/3) * self.T**2 * (np.transpose(self.A_cont, (1, 0))*self.dQ)

        self.dQ = self.dQ + M2 + M3

    def kf_update_loop(self, y_tilde):
        self.y_tilde = y_tilde
        # print('y_tilde: ', self.y_tilde)
        # print('y: ', self.y)
        self.y = self.integrate(self.y)
        self.linearize(self.y)
        self.discretizeQ()

        A_t = np.transpose(self.A, (1, 0))
        # print('A: ', self.A)
        # print('A_t: ', A_t)
        self.P = np.matmul(np.matmul(self.A, self.P), A_t) + self.dQ
        # print('P left: ', self.P_left)
        # pt1 =
        # pt2 =
        # print('pt1: ', pt1)
        # print('pt2: ', pt2)
        self.G = np.matmul(np.matmul(self.P, np.transpose(self.H, (1, 0))),
                           np.linalg.inv(np.matmul(np.matmul(self.H, self.P), np.transpose(self.H, (1, 0))) + self.R))

        self.y = self.y + np.matmul(self.G, y_tilde - np.matmul(self.H, self.y))
        self.P = np.matmul((np.eye(5) - np.matmul(self.G, self.H)), self.P)

        print('P after update: ', self.P)
        # print('P_left: ', self.P_left)
        # print('p_right: ', self.P_right)

        if np.linalg.norm(self.P) > 1000:
            print('hard restarting model')
            self.P = np.array([[10.0e-6, 0.0, 0.0, 0.0], [0.0, 10.0e-4, 0.0, 0.0],
                                    [0.0, 0.0, 10.0e-3, 0.0], [0.0, 0.0, 0.0, 10e-4]], dtype=float)  # covariance matrix
            self.y = np.array([[1.0], [1.0], [0.0], [0.0]], dtype=float)  # MP state

        self.t0 = self.t

