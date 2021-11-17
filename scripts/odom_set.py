import numpy as np


def odom_set(odom_msg, first_call, theta_0, x_0, y_0):


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

    return x, y, theta, first_call
