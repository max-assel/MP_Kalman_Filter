import numpy as np
from MP_KF import MPKalmanFilter

class Gap:
    def __init__(self, left_bearing, left_range, right_bearing, right_range, transform):
        self.left_bearing_deg = left_bearing  # in degrees
        self.left_bearing_rad = self.left_bearing_deg * np.pi / 180.0
        self.left_range = left_range
        self.right_bearing_deg = right_bearing  # in degrees
        self.right_bearing_rad = self.right_bearing_deg * np.pi / 180.0
        self.right_range = right_range
        self.transform = transform

        self.left_model = MPKalmanFilter()
        self.right_model = MPKalmanFilter()

    def set_left_model(self, model):
        self.left_model = model

    def set_right_model(self, model):
        self.right_model = model

    def get_left_gap_world(self):
        left_pt_world = np.matmul(self.transform, np.array([[self.left_range*np.cos(self.left_bearing_rad)],
                                                            [self.left_range*np.sin(self.left_bearing_rad)],
                                                            [1]], dtype=float))
        return left_pt_world

    def get_right_gap_world(self):
        right_pt_world = np.matmul(self.transform, np.array([[self.right_range * np.cos(self.right_bearing_rad)],
                                                            [self.right_range * np.sin(self.right_bearing_rad)],
                                                            [1]], dtype=float))
        return right_pt_world

    def get_left_gap_point_cartesian(self, affine=False):
        if affine:
            left_pt = np.array([[self.left_range*np.cos(self.left_bearing_rad)],
                                [self.left_range*np.sin(self.left_bearing_rad)],
                                [1]], dtype=float)
        else:
            left_pt = np.array([[self.left_range*np.cos(self.left_bearing_rad)],
                                [self.left_range*np.sin(self.left_bearing_rad)]], dtype=float)
        # print('left_pt: ', left_pt)
        return left_pt

    def get_right_gap_point_cartesian(self, affine=False):
        if affine:
            right_pt = np.array([[self.right_range*np.cos(self.right_bearing_rad)],
                                 [self.right_range*np.sin(self.right_bearing_rad)],
                                 [1]], dtype=float)
        else:
            right_pt = np.array([[self.right_range*np.cos(self.right_bearing_rad)],
                                 [self.right_range*np.sin(self.right_bearing_rad)]], dtype=float)
        # print('right_pt: ', right_pt)
        return right_pt

