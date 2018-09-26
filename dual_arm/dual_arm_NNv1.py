#!/usr/bin/env python
# coding=utf-8

import math
import numpy as np
import baxter_interface
from baxter_interface import CHECK_VERSION
import rospy

import matplotlib.pyplot as plt
import threading
from mpl_toolkits.mplot3d import Axes3D
import random


size1 = 20
size2 = 18
linesize = 2

# legend 字体
legend_size = 16

# 标题字体
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': size1,
         }
# 坐标轴刻度和坐标轴名称
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': size2,
         }

def left_arm_control():
    """ 获取期望关节角和速度 """
    Rate = rospy.Rate(200)
    left_arm = baxter_interface.Limb("left")

    joint_angles_goal_list = [[-1.05, 1.245, -0.677, -1.134, 2.358, 0.472, -0.935],
                              [-1.05, 1.246, -0.678, -1.133, 2.357, 0.471, -0.934],
                              [-1.05, 1.246, -0.68, -1.133, 2.354, 0.47, -0.933],
                              [-1.049, 1.248, -0.683, -1.131, 2.349, 0.467, -0.931],
                              [-1.048, 1.25, -0.687, -1.13, 2.343, 0.464, -0.928],
                              [-1.046, 1.252, -0.692, -1.128, 2.335, 0.46, -0.924],
                              [-1.045, 1.254, -0.698, -1.125, 2.325, 0.456, -0.919],
                              [-1.043, 1.258, -0.704, -1.123, 2.314, 0.451, -0.914],
                              [-1.041, 1.261, -0.712, -1.12, 2.301, 0.445, -0.908],
                              [-1.039, 1.264, -0.72, -1.116, 2.288, 0.438, -0.902],
                              [-1.036, 1.268, -0.728, -1.113, 2.273, 0.431, -0.896],
                              [-1.034, 1.272, -0.737, -1.109, 2.258, 0.424, -0.889],
                              [-1.031, 1.276, -0.745, -1.105, 2.242, 0.416, -0.882],
                              [-1.029, 1.281, -0.755, -1.101, 2.225, 0.408, -0.875],
                              [-1.026, 1.285, -0.764, -1.097, 2.208, 0.4, -0.867],
                              [-1.024, 1.289, -0.773, -1.093, 2.19, 0.391, -0.86],
                              [-1.021, 1.294, -0.782, -1.089, 2.172, 0.382, -0.853],
                              [-1.019, 1.298, -0.791, -1.085, 2.154, 0.374, -0.846],
                              [-1.016, 1.302, -0.799, -1.081, 2.136, 0.365, -0.839],
                              [-1.014, 1.306, -0.807, -1.076, 2.118, 0.356, -0.832],
                              [-1.012, 1.31, -0.815, -1.072, 2.1, 0.347, -0.826],
                              [-1.009, 1.314, -0.823, -1.068, 2.082, 0.338, -0.819],
                              [-1.007, 1.318, -0.831, -1.064, 2.064, 0.329, -0.813],
                              [-1.005, 1.322, -0.838, -1.06, 2.046, 0.32, -0.807],
                              [-1.003, 1.326, -0.845, -1.056, 2.028, 0.31, -0.801],
                              [-1.001, 1.33, -0.853, -1.052, 2.01, 0.301, -0.795],
                              [-0.999, 1.334, -0.86, -1.048, 1.992, 0.292, -0.789],
                              [-0.997, 1.337, -0.867, -1.044, 1.975, 0.283, -0.783],
                              [-0.995, 1.341, -0.873, -1.04, 1.957, 0.274, -0.777],
                              [-0.993, 1.345, -0.88, -1.036, 1.939, 0.265, -0.771],
                              [-0.991, 1.348, -0.887, -1.032, 1.921, 0.256, -0.765],
                              [-0.99, 1.352, -0.894, -1.028, 1.903, 0.246, -0.759],
                              [-0.988, 1.355, -0.901, -1.024, 1.886, 0.237, -0.753],
                              [-0.986, 1.359, -0.908, -1.02, 1.868, 0.228, -0.747],
                              [-0.984, 1.363, -0.915, -1.016, 1.851, 0.22, -0.741],
                              [-0.982, 1.366, -0.922, -1.012, 1.833, 0.211, -0.735],
                              [-0.98, 1.37, -0.929, -1.008, 1.816, 0.202, -0.729],
                              [-0.978, 1.374, -0.936, -1.004, 1.798, 0.193, -0.723],
                              [-0.976, 1.377, -0.943, -1.0, 1.781, 0.184, -0.716],
                              [-0.974, 1.381, -0.951, -0.996, 1.764, 0.176, -0.71],
                              [-0.972, 1.385, -0.958, -0.991, 1.747, 0.167, -0.704],
                              [-0.97, 1.388, -0.965, -0.987, 1.73, 0.158, -0.697],
                              [-0.968, 1.392, -0.973, -0.983, 1.713, 0.15, -0.691],
                              [-0.966, 1.395, -0.98, -0.979, 1.696, 0.141, -0.685],
                              [-0.964, 1.399, -0.987, -0.975, 1.679, 0.133, -0.678],
                              [-0.962, 1.403, -0.995, -0.971, 1.662, 0.124, -0.672],
                              [-0.96, 1.406, -1.002, -0.967, 1.645, 0.116, -0.666],
                              [-0.958, 1.41, -1.01, -0.963, 1.628, 0.107, -0.659],
                              [-0.955, 1.414, -1.017, -0.959, 1.611, 0.099, -0.653],
                              [-0.953, 1.417, -1.024, -0.955, 1.595, 0.09, -0.647],
                              [-0.951, 1.421, -1.032, -0.951, 1.578, 0.082, -0.64],
                              [-0.949, 1.424, -1.039, -0.948, 1.561, 0.073, -0.634],
                              [-0.947, 1.428, -1.046, -0.944, 1.545, 0.065, -0.628],
                              [-0.944, 1.432, -1.054, -0.94, 1.528, 0.056, -0.622],
                              [-0.942, 1.435, -1.061, -0.936, 1.512, 0.048, -0.616],
                              [-0.94, 1.439, -1.068, -0.932, 1.495, 0.039, -0.61],
                              [-0.937, 1.442, -1.075, -0.929, 1.479, 0.031, -0.604],
                              [-0.935, 1.446, -1.082, -0.925, 1.462, 0.022, -0.598],
                              [-0.933, 1.45, -1.09, -0.921, 1.446, 0.014, -0.592],
                              [-0.93, 1.453, -1.097, -0.917, 1.43, 0.005, -0.586],
                              [-0.928, 1.457, -1.104, -0.914, 1.413, -0.003, -0.58],
                              [-0.925, 1.46, -1.111, -0.91, 1.397, -0.011, -0.574],
                              [-0.923, 1.464, -1.118, -0.906, 1.381, -0.019, -0.568],
                              [-0.921, 1.467, -1.126, -0.903, 1.365, -0.028, -0.563],
                              [-0.918, 1.471, -1.133, -0.899, 1.349, -0.036, -0.557],
                              [-0.916, 1.475, -1.14, -0.895, 1.333, -0.044, -0.551],
                              [-0.914, 1.478, -1.147, -0.891, 1.317, -0.052, -0.545],
                              [-0.912, 1.482, -1.154, -0.888, 1.301, -0.059, -0.539],
                              [-0.909, 1.486, -1.162, -0.884, 1.286, -0.067, -0.533],
                              [-0.907, 1.489, -1.169, -0.88, 1.27, -0.075, -0.527],
                              [-0.905, 1.493, -1.176, -0.876, 1.254, -0.082, -0.521],
                              [-0.903, 1.496, -1.184, -0.873, 1.238, -0.09, -0.515],
                              [-0.901, 1.5, -1.191, -0.869, 1.223, -0.098, -0.509],
                              [-0.898, 1.504, -1.198, -0.865, 1.207, -0.105, -0.503],
                              [-0.896, 1.507, -1.205, -0.861, 1.191, -0.113, -0.497],
                              [-0.894, 1.511, -1.213, -0.857, 1.175, -0.121, -0.491],
                              [-0.892, 1.514, -1.22, -0.853, 1.159, -0.128, -0.485],
                              [-0.89, 1.518, -1.227, -0.85, 1.143, -0.136, -0.479],
                              [-0.888, 1.521, -1.234, -0.846, 1.126, -0.144, -0.473],
                              [-0.886, 1.525, -1.241, -0.842, 1.11, -0.152, -0.466],
                              [-0.884, 1.528, -1.248, -0.838, 1.093, -0.161, -0.46],
                              [-0.882, 1.531, -1.255, -0.834, 1.076, -0.169, -0.454],
                              [-0.88, 1.535, -1.262, -0.83, 1.059, -0.178, -0.447],
                              [-0.878, 1.538, -1.269, -0.826, 1.041, -0.187, -0.441],
                              [-0.877, 1.541, -1.276, -0.821, 1.023, -0.196, -0.434],
                              [-0.875, 1.544, -1.282, -0.817, 1.005, -0.205, -0.427],
                              [-0.873, 1.547, -1.289, -0.813, 0.987, -0.215, -0.42],
                              [-0.871, 1.55, -1.295, -0.809, 0.969, -0.224, -0.414],
                              [-0.869, 1.553, -1.302, -0.805, 0.951, -0.233, -0.407],
                              [-0.868, 1.555, -1.307, -0.801, 0.934, -0.243, -0.401],
                              [-0.866, 1.558, -1.313, -0.798, 0.917, -0.251, -0.395],
                              [-0.865, 1.56, -1.318, -0.794, 0.902, -0.26, -0.389],
                              [-0.863, 1.562, -1.323, -0.791, 0.887, -0.268, -0.384],
                              [-0.862, 1.564, -1.327, -0.788, 0.873, -0.275, -0.379],
                              [-0.861, 1.566, -1.331, -0.785, 0.86, -0.282, -0.374],
                              [-0.86, 1.568, -1.335, -0.783, 0.849, -0.288, -0.37],
                              [-0.859, 1.569, -1.338, -0.781, 0.84, -0.293, -0.367],
                              [-0.859, 1.57, -1.34, -0.779, 0.833, -0.297, -0.364],
                              [-0.858, 1.571, -1.342, -0.778, 0.827, -0.3, -0.362],
                              [-0.858, 1.571, -1.343, -0.777, 0.823, -0.302, -0.361],
                              [-0.858, 1.571, -1.343, -0.777, 0.822, -0.303, -0.36]
                              ]
    joint_vel_goal_list = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.012, 0.019, -0.043, 0.017, -0.069, -0.032, 0.033],
                           [0.023, 0.037, -0.083, 0.033, -0.133, -0.062, 0.064],
                           [0.033, 0.054, -0.119, 0.048, -0.194, -0.091, 0.092],
                           [0.043, 0.069, -0.152, 0.061, -0.25, -0.118, 0.118],
                           [0.051, 0.082, -0.182, 0.074, -0.302, -0.142, 0.141],
                           [0.059, 0.095, -0.209, 0.085, -0.35, -0.165, 0.162],
                           [0.065, 0.106, -0.232, 0.096, -0.394, -0.186, 0.18],
                           [0.071, 0.116, -0.252, 0.105, -0.433, -0.206, 0.197],
                           [0.076, 0.124, -0.269, 0.113, -0.468, -0.223, 0.21],
                           [0.08, 0.131, -0.283, 0.12, -0.499, -0.238, 0.221],
                           [0.083, 0.136, -0.294, 0.126, -0.526, -0.252, 0.23],
                           [0.085, 0.141, -0.301, 0.131, -0.549, -0.264, 0.237],
                           [0.086, 0.144, -0.305, 0.135, -0.567, -0.274, 0.241],
                           [0.086, 0.145, -0.305, 0.137, -0.581, -0.282, 0.242],
                           [0.085, 0.145, -0.303, 0.139, -0.591, -0.289, 0.241],
                           [0.084, 0.144, -0.297, 0.139, -0.597, -0.293, 0.238],
                           [0.081, 0.141, -0.288, 0.139, -0.598, -0.296, 0.232],
                           [0.079, 0.139, -0.279, 0.138, -0.599, -0.298, 0.227],
                           [0.076, 0.136, -0.271, 0.137, -0.6, -0.3, 0.221],
                           [0.074, 0.134, -0.263, 0.136, -0.6, -0.301, 0.217],
                           [0.072, 0.132, -0.256, 0.136, -0.6, -0.303, 0.212],
                           [0.07, 0.13, -0.25, 0.135, -0.6, -0.304, 0.209],
                           [0.069, 0.128, -0.245, 0.135, -0.6, -0.304, 0.206],
                           [0.067, 0.127, -0.24, 0.134, -0.599, -0.305, 0.203],
                           [0.066, 0.125, -0.236, 0.134, -0.598, -0.305, 0.201],
                           [0.065, 0.124, -0.233, 0.134, -0.597, -0.305, 0.199],
                           [0.064, 0.123, -0.231, 0.134, -0.596, -0.305, 0.198],
                           [0.064, 0.122, -0.229, 0.134, -0.595, -0.305, 0.197],
                           [0.063, 0.122, -0.228, 0.134, -0.594, -0.304, 0.197],
                           [0.063, 0.121, -0.228, 0.134, -0.592, -0.303, 0.197],
                           [0.063, 0.121, -0.228, 0.134, -0.59, -0.302, 0.198],
                           [0.063, 0.121, -0.229, 0.134, -0.588, -0.3, 0.199],
                           [0.063, 0.121, -0.231, 0.135, -0.586, -0.298, 0.201],
                           [0.064, 0.121, -0.234, 0.135, -0.583, -0.296, 0.203],
                           [0.064, 0.121, -0.236, 0.135, -0.581, -0.295, 0.205],
                           [0.065, 0.121, -0.238, 0.136, -0.579, -0.293, 0.207],
                           [0.065, 0.121, -0.24, 0.136, -0.577, -0.291, 0.208],
                           [0.066, 0.121, -0.241, 0.136, -0.575, -0.29, 0.209],
                           [0.066, 0.122, -0.243, 0.136, -0.573, -0.289, 0.21],
                           [0.067, 0.122, -0.244, 0.136, -0.571, -0.288, 0.211],
                           [0.068, 0.122, -0.245, 0.135, -0.569, -0.287, 0.212],
                           [0.068, 0.122, -0.246, 0.135, -0.567, -0.286, 0.212],
                           [0.069, 0.121, -0.246, 0.135, -0.565, -0.285, 0.212],
                           [0.07, 0.121, -0.247, 0.134, -0.564, -0.285, 0.212],
                           [0.07, 0.121, -0.247, 0.134, -0.562, -0.284, 0.212],
                           [0.071, 0.121, -0.247, 0.133, -0.561, -0.284, 0.211],
                           [0.072, 0.121, -0.247, 0.132, -0.559, -0.284, 0.21],
                           [0.073, 0.121, -0.246, 0.132, -0.558, -0.284, 0.209],
                           [0.074, 0.121, -0.245, 0.131, -0.557, -0.284, 0.208],
                           [0.075, 0.12, -0.245, 0.13, -0.556, -0.284, 0.207],
                           [0.075, 0.12, -0.244, 0.129, -0.554, -0.284, 0.205],
                           [0.076, 0.12, -0.243, 0.128, -0.553, -0.284, 0.204],
                           [0.077, 0.12, -0.242, 0.127, -0.552, -0.284, 0.202],
                           [0.077, 0.119, -0.241, 0.126, -0.551, -0.283, 0.201],
                           [0.078, 0.119, -0.241, 0.126, -0.549, -0.283, 0.2],
                           [0.078, 0.119, -0.24, 0.125, -0.548, -0.282, 0.199],
                           [0.078, 0.119, -0.24, 0.124, -0.546, -0.281, 0.198],
                           [0.079, 0.119, -0.24, 0.124, -0.544, -0.28, 0.197],
                           [0.079, 0.119, -0.24, 0.124, -0.543, -0.278, 0.197],
                           [0.079, 0.119, -0.239, 0.124, -0.541, -0.277, 0.196],
                           [0.078, 0.119, -0.24, 0.123, -0.539, -0.275, 0.196],
                           [0.078, 0.12, -0.24, 0.123, -0.537, -0.273, 0.196],
                           [0.078, 0.12, -0.24, 0.123, -0.535, -0.27, 0.196],
                           [0.077, 0.12, -0.24, 0.124, -0.533, -0.268, 0.195],
                           [0.077, 0.12, -0.241, 0.124, -0.531, -0.265, 0.196],
                           [0.076, 0.12, -0.241, 0.124, -0.528, -0.262, 0.196],
                           [0.075, 0.121, -0.242, 0.124, -0.526, -0.259, 0.196],
                           [0.074, 0.121, -0.242, 0.125, -0.524, -0.257, 0.197],
                           [0.074, 0.121, -0.243, 0.125, -0.523, -0.255, 0.197],
                           [0.073, 0.121, -0.243, 0.126, -0.523, -0.254, 0.198],
                           [0.072, 0.121, -0.243, 0.126, -0.524, -0.253, 0.199],
                           [0.071, 0.12, -0.243, 0.127, -0.525, -0.253, 0.2],
                           [0.07, 0.12, -0.242, 0.128, -0.527, -0.254, 0.201],
                           [0.07, 0.119, -0.241, 0.128, -0.53, -0.256, 0.202],
                           [0.069, 0.118, -0.241, 0.129, -0.534, -0.258, 0.204],
                           [0.068, 0.117, -0.24, 0.13, -0.538, -0.261, 0.205],
                           [0.067, 0.116, -0.238, 0.131, -0.544, -0.265, 0.207],
                           [0.067, 0.115, -0.237, 0.132, -0.55, -0.269, 0.209],
                           [0.066, 0.113, -0.236, 0.132, -0.556, -0.274, 0.211],
                           [0.065, 0.112, -0.234, 0.133, -0.564, -0.28, 0.213],
                           [0.064, 0.11, -0.232, 0.134, -0.572, -0.286, 0.216],
                           [0.064, 0.108, -0.23, 0.135, -0.581, -0.293, 0.218],
                           [0.063, 0.106, -0.227, 0.137, -0.591, -0.301, 0.221],
                           [0.062, 0.104, -0.225, 0.137, -0.601, -0.308, 0.223],
                           [0.061, 0.101, -0.221, 0.137, -0.605, -0.313, 0.224],
                           [0.06, 0.098, -0.215, 0.136, -0.603, -0.314, 0.222],
                           [0.058, 0.094, -0.209, 0.133, -0.596, -0.313, 0.219],
                           [0.055, 0.09, -0.201, 0.13, -0.583, -0.308, 0.214],
                           [0.053, 0.085, -0.191, 0.125, -0.565, -0.3, 0.207],
                           [0.05, 0.08, -0.18, 0.119, -0.542, -0.288, 0.198],
                           [0.047, 0.074, -0.168, 0.113, -0.512, -0.274, 0.187],
                           [0.043, 0.068, -0.155, 0.104, -0.478, -0.256, 0.174],
                           [0.039, 0.061, -0.14, 0.095, -0.437, -0.235, 0.159],
                           [0.034, 0.054, -0.124, 0.085, -0.392, -0.211, 0.142],
                           [0.03, 0.046, -0.107, 0.074, -0.34, -0.184, 0.123],
                           [0.024, 0.038, -0.088, 0.061, -0.283, -0.154, 0.102],
                           [0.019, 0.029, -0.068, 0.048, -0.221, -0.12, 0.08],
                           [0.013, 0.02, -0.047, 0.033, -0.153, -0.083, 0.055],
                           [0.007, 0.01, -0.024, 0.017, -0.079, -0.043, 0.028],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                           ]
    """ 初始化变量 """
    point_sum = len(joint_angles_goal_list)
    point_now = 0
    joint_angles_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_velocities_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # 速度误差
    z2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dy_tau = left_arm.joint_efforts()    # 实际关节转矩

    count = 2000    # 迭代次数
    output_size = 100
    out_ratio = count / output_size  # 作图抽取率

    """ 画图用初始化，输入关节力，关节实际位置，关节期望位置，时间，末端实际位置，末端期望位置 """
    joint_effort_display = np.zeros((7, output_size + 1), dtype=float)
    joint_actual_pos_display = np.zeros((7, output_size + 1), dtype=float)
    joint_req_pos_display = np.zeros((7, output_size + 1), dtype=float)
    tout = np.zeros((7, output_size + 1), dtype=float)
    xyz_display = np.zeros((3, output_size + 1), dtype=float)
    xyz_req_display = np.zeros((3, output_size + 1), dtype=float)
    z3_display = np.zeros((7, output_size + 1), dtype=float)
    sat_display = np.zeros((1, output_size + 1), dtype=float)

    """ 产生高斯白噪声 """
    mu = 0
    sigma = 0.3
    f_dis = z3_display = np.zeros((7, output_size + 1), dtype=float)

    a = 0  # 输出用计数

    for i in range(0, count):
        if not rospy.is_shutdown():   # 循环过程中ctrl+c判断
            """ 得到角度,把字典形式写成序列形式 """
            joint_angles_now = left_arm.joint_angles()
            temp = 0
            for key in joint_angles_now:
                joint_angles_now_list[temp] = joint_angles_now[key]
                temp = temp + 1

            """ 得到速度,把字典形式写成序列形式 """
            joint_velocities_now = left_arm.joint_velocities()
            temp = 0
            for key in joint_velocities_now:
                joint_velocities_now_list[temp] = joint_velocities_now[key]
                temp = temp + 1

            """ 计算出当前应该的目标点 """
            if point_now < point_sum:   # point_sum等于100，每8个迭代点算一次目标值
                if i % 8 == 0:
                    point_now = point_now + 1

            """ 计算z1,计算关节角误差，上面8个+1，因此这里计算当前-1 """
            for aaa in range(0, 7):
                z1[aaa] = joint_angles_now_list[aaa] - joint_angles_goal_list[point_now - 1][aaa]

            """ 计算alpha """
            for aaa in range(0, 7):
                alpha[aaa] = -K1_L[aaa] * z1[aaa] + joint_vel_goal_list[point_now - 1][aaa]

            """ 计算z2 """
            for aaa in range(0, 7):
                z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]

            """ 得到通信计算的值，即神经网络部分的值,udp_l.controller.torController[temp].get_kd()只是7个数 """
            temp = 0
            for key in dy_tau:
                dy_tau[key] = -z1[temp] - K2_L[aaa] * z2[temp]
                # if dy_tau[key] > sat:
                #     dy_tau[key] = sat
                # if dy_tau[key] < -1 * sat:
                #     dy_tau[key] = -1 * sat
                temp = temp + 1

            """ 画图初始化 """
            sat = 2.0
            if a == 0:
                temp = 0
                xyz_pose = left_arm.endpoint_pose()
                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.4
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4

                start_time = rospy.get_time()
                get_pose = left_arm.joint_angles()
                z3_display[temp, a] = 0
                sat_display[0, a] = sat
                # 把字典转换成序列
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]

                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now - 1][temp]
                    tout[temp, a] = 0
                    temp = temp + 1
                a = a + 1
            # print point_now

            left_arm.set_joint_torques(dy_tau)

            """ 作图用 """
            if i % out_ratio == 0:
                display_cur_time = rospy.get_time() # 控制后的时间
                """ 关节角度 """
                temp = 0
                get_pose = left_arm.joint_angles()
                xyz_pose = left_arm.endpoint_pose()

                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.4 + 0.004 * a
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4
                sat_display[0, a] = sat
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    if joint_effort_display[temp, a] > sat:
                        joint_effort_display[temp, a] = sat
                    if joint_effort_display[temp, a] < -1 * sat:
                        joint_effort_display[temp, a] = -1 * sat

                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now - 1][temp]
                    tout[temp, a] = display_cur_time - start_time

                    """ 计算扰动 """
                    f_dis = 2 * random.gauss(mu, sigma)
                    deta_tau = joint_effort_display[temp, a] - dy_tau[key]
                    K = 2
                    D = -f_dis + deta_tau
                    z3_display[temp, a] = D - K * z2[temp]

                    temp = temp + 1

                a = a + 1
            Rate.sleep()
        # rospy.on_shutdown(True)
    left_arm.exit_control_mode()

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    plt.title("Left joint W0", font1)
    lns1 = ax.plot(tout[0].T, joint_actual_pos_display[0], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[0].T, joint_req_pos_display[0], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[0] - joint_actual_pos_display[0], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig1.tight_layout()
    plt.savefig('picture/left/left_joint_w0.eps', format='eps')

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    plt.title("Left joint W1", font1)
    lns1 = ax.plot(tout[1].T, joint_actual_pos_display[1], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[1].T, joint_req_pos_display[1], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[1] - joint_actual_pos_display[1], '-*', linewidth=3, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig2.tight_layout()
    plt.savefig('picture/left/left_joint_w1.eps', format='eps')

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    plt.title("Left joint W2", font1)
    lns1 = ax.plot(tout[2].T, joint_actual_pos_display[2], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[2].T, joint_req_pos_display[2], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[2] - joint_actual_pos_display[2], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig3.tight_layout()
    plt.savefig('picture/left/left_joint_w2.eps', format='eps')

    fig4 = plt.figure(4)
    ax = fig4.add_subplot(111)
    plt.title("Left joint E0", font1)
    lns1 = ax.plot(tout[3].T, joint_actual_pos_display[3], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[3].T, joint_req_pos_display[3], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[3] - joint_actual_pos_display[3], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig4.tight_layout()
    plt.savefig('picture/left/left_joint_e0.eps', format='eps')

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    plt.title("Left joint E1", font1)
    lns1 = ax.plot(tout[4].T, joint_actual_pos_display[4], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[4].T, joint_req_pos_display[4], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[4] - joint_actual_pos_display[4], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig5.tight_layout()
    plt.savefig('picture/left/left_joint_e1.eps', format='eps')

    fig6 = plt.figure(6)
    ax = fig6.add_subplot(111)
    plt.title("Left joint S0", font1)
    lns1 = ax.plot(tout[5].T, joint_actual_pos_display[5], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[5].T, joint_req_pos_display[5], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[5] - joint_actual_pos_display[5], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig6.tight_layout()
    plt.savefig('picture/left/left_joint_s0.eps', format='eps')

    fig7 = plt.figure(7)
    ax = fig7.add_subplot(111)
    plt.title("Left joint S1", font1)
    lns1 = ax.plot(tout[6].T, joint_actual_pos_display[6], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[6].T, joint_req_pos_display[6], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[6] - joint_actual_pos_display[6], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig7.tight_layout()
    plt.savefig('picture/left/left_joint_s1.eps', format='eps')

    fig8 = plt.figure(8)
    ax = fig8.add_subplot(111)
    plt.title("Left torques")
    plt.plot(tout[0].T, joint_effort_display[0], '-*', linewidth=linesize, label='Joint W0')
    plt.plot(tout[0].T, joint_effort_display[1], '-o', linewidth=linesize, label='Joint W1')
    plt.plot(tout[0].T, joint_effort_display[2], '-x', linewidth=linesize, label='Joint W2')
    plt.plot(tout[0].T, joint_effort_display[3], '-s', linewidth=linesize, label='Joint E0')
    plt.plot(tout[0].T, joint_effort_display[4], '-p', linewidth=linesize, label='Joint E1')
    plt.plot(tout[0].T, joint_effort_display[5], '-h', linewidth=linesize, label='Joint S0')
    plt.plot(tout[0].T, joint_effort_display[6], '-d', linewidth=linesize, label='Joint S1')
    plt.plot(tout[0].T, sat_display[0], 'm--', linewidth=linesize, label='Limit value')
    plt.plot(tout[0].T, -1*sat_display[0], 'm--', linewidth=linesize)
    plt.xlabel("Time/s", font2)
    plt.ylabel("Torque/Nm", font2)
    ax.grid()
    plt.xlim(0, 7.3)
    plt.ylim(-2.5, 2.5)
    # plt.legend(bbox_to_anchor=(1.105, 0.6), fontsize=legend_size)
    plt.legend(loc=7, fontsize=legend_size)
    fig8.tight_layout()
    plt.savefig('picture/left/left_torques.eps', format='eps')

    fig9 = plt.figure(9)
    ax = fig9.add_subplot(111, projection='3d')
    ax.grid()
    plt.title("Left endpoint position")
    ax.plot(xyz_display[0], xyz_display[1], xyz_display[2], '-o', label="Actual value", color='red', linewidth=linesize)
    ax.plot(xyz_req_display[0], xyz_req_display[1], xyz_req_display[2],'-*', label="Desired value", color='green', linewidth=linesize)
    ax.set_xlabel("X/m", font2)
    ax.set_ylabel("Y/m", font2)
    ax.set_zlabel("Z/m", font2)
    plt.ylim(-1, 1)
    ax.set_zlim(0.2, 0.5)
    # ax.set_xlim(0, 1)
    fig9.tight_layout()
    # plt.legend(loc='best', bbox_to_anchor=(1.1, 1), fontsize=legend_size)
    plt.legend(loc=0, fontsize=legend_size)
    plt.savefig('picture/left/left_endpoint_position.eps', format='eps')

    fig10 = plt.figure(10)
    ax = fig10.add_subplot(111)
    ax.grid()
    plt.title("Left z3")
    plt.plot(tout[0].T, z3_display[0], '-*', linewidth=linesize, label='Joint W0')
    plt.plot(tout[0].T, z3_display[1], '-o', linewidth=linesize, label='Joint W1')
    plt.plot(tout[0].T, z3_display[2], '-x', linewidth=linesize, label='Joint W2')
    plt.plot(tout[0].T, z3_display[3], '-s', linewidth=linesize, label='Joint E0')
    plt.plot(tout[0].T, z3_display[4], '-p', linewidth=linesize, label='Joint E1')
    plt.plot(tout[0].T, z3_display[5], '-h', linewidth=linesize, label='Joint S0')
    plt.plot(tout[0].T, z3_display[6], '-d', linewidth=linesize, label='Joint S1')
    plt.xlabel("Time/s", font2)
    plt.ylabel("Torque/Nm", font2)
    plt.xlim(0, 7.3)
    # plt.ylim(-4, 4)
    # plt.legend(bbox_to_anchor=(1.095, 0.6), fontsize=legend_size)
    plt.legend(loc=7, fontsize=legend_size)
    fig10.tight_layout()
    plt.savefig('picture/left/left_z3.eps', format='eps')

    # plt.show()

def right_arm_control():
    """ 获取期望关节角和速度 """
    Rate = rospy.Rate(200)
    right_arm = baxter_interface.Limb("right")

    joint_angles_goal_list = [
        [-0.397, -1.181, 1.204, 1.4, 0.556, 0.991, 2.436],
        [-0.396, -1.18, 1.204, 1.4, 0.556, 0.991, 2.434],
        [-0.395, -1.179, 1.203, 1.4, 0.559, 0.99, 2.431],
        [-0.393, -1.176, 1.202, 1.401, 0.562, 0.989, 2.426],
        [-0.389, -1.172, 1.201, 1.402, 0.567, 0.987, 2.419],
        [-0.386, -1.168, 1.2, 1.403, 0.572, 0.985, 2.41],
        [-0.381, -1.162, 1.198, 1.404, 0.579, 0.982, 2.399],
        [-0.376, -1.156, 1.196, 1.405, 0.586, 0.979, 2.388],
        [-0.37, -1.15, 1.194, 1.407, 0.595, 0.976, 2.374],
        [-0.363, -1.143, 1.192, 1.408, 0.603, 0.972, 2.36],
        [-0.356, -1.135, 1.19, 1.41, 0.613, 0.968, 2.345],
        [-0.349, -1.127, 1.187, 1.412, 0.622, 0.964, 2.328],
        [-0.341, -1.119, 1.185, 1.414, 0.632, 0.96, 2.312],
        [-0.333, -1.11, 1.182, 1.416, 0.642, 0.956, 2.294],
        [-0.325, -1.101, 1.18, 1.417, 0.653, 0.951, 2.276],
        [-0.317, -1.092, 1.177, 1.419, 0.663, 0.947, 2.258],
        [-0.309, -1.083, 1.174, 1.421, 0.673, 0.942, 2.24],
        [-0.3, -1.075, 1.171, 1.423, 0.682, 0.938, 2.221],
        [-0.292, -1.066, 1.169, 1.425, 0.692, 0.933, 2.203],
        [-0.283, -1.057, 1.166, 1.427, 0.701, 0.929, 2.185],
        [-0.275, -1.049, 1.164, 1.428, 0.709, 0.924, 2.167],
        [-0.266, -1.041, 1.161, 1.43, 0.718, 0.92, 2.149],
        [-0.258, -1.032, 1.159, 1.432, 0.726, 0.915, 2.131],
        [-0.249, -1.024, 1.156, 1.433, 0.734, 0.911, 2.113],
        [-0.241, -1.016, 1.154, 1.435, 0.742, 0.907, 2.095],
        [-0.232, -1.008, 1.151, 1.436, 0.75, 0.902, 2.077],
        [-0.224, -1.0, 1.149, 1.438, 0.757, 0.898, 2.059],
        [-0.215, -0.992, 1.146, 1.439, 0.765, 0.894, 2.042],
        [-0.207, -0.984, 1.144, 1.441, 0.772, 0.89, 2.024],
        [-0.198, -0.976, 1.141, 1.442, 0.779, 0.885, 2.007],
        [-0.19, -0.968, 1.139, 1.444, 0.787, 0.881, 1.989],
        [-0.182, -0.96, 1.137, 1.445, 0.794, 0.877, 1.972],
        [-0.173, -0.952, 1.134, 1.447, 0.801, 0.873, 1.954],
        [-0.165, -0.944, 1.132, 1.448, 0.809, 0.868, 1.937],
        [-0.156, -0.936, 1.129, 1.449, 0.816, 0.864, 1.919],
        [-0.148, -0.928, 1.127, 1.451, 0.824, 0.86, 1.902],
        [-0.139, -0.92, 1.124, 1.452, 0.831, 0.856, 1.885],
        [-0.131, -0.912, 1.122, 1.454, 0.839, 0.852, 1.867],
        [-0.122, -0.904, 1.12, 1.455, 0.847, 0.848, 1.85],
        [-0.114, -0.896, 1.117, 1.457, 0.855, 0.843, 1.833],
        [-0.105, -0.888, 1.115, 1.458, 0.863, 0.839, 1.816],
        [-0.097, -0.88, 1.112, 1.46, 0.871, 0.835, 1.798],
        [-0.089, -0.872, 1.11, 1.461, 0.879, 0.831, 1.781],
        [-0.08, -0.864, 1.107, 1.463, 0.887, 0.827, 1.764],
        [-0.072, -0.856, 1.105, 1.464, 0.895, 0.823, 1.748],
        [-0.064, -0.849, 1.103, 1.466, 0.903, 0.819, 1.731],
        [-0.055, -0.841, 1.1, 1.467, 0.911, 0.815, 1.714],
        [-0.047, -0.833, 1.098, 1.468, 0.919, 0.811, 1.697],
        [-0.039, -0.825, 1.095, 1.47, 0.927, 0.807, 1.681],
        [-0.03, -0.818, 1.093, 1.471, 0.935, 0.802, 1.664],
        [-0.022, -0.81, 1.091, 1.473, 0.942, 0.798, 1.648],
        [-0.014, -0.802, 1.088, 1.474, 0.95, 0.794, 1.632],
        [-0.006, -0.795, 1.086, 1.475, 0.958, 0.79, 1.615],
        [0.002, -0.787, 1.084, 1.477, 0.965, 0.786, 1.599],
        [0.01, -0.78, 1.081, 1.478, 0.973, 0.782, 1.583],
        [0.019, -0.773, 1.079, 1.48, 0.98, 0.778, 1.567],
        [0.027, -0.765, 1.076, 1.481, 0.988, 0.774, 1.551],
        [0.035, -0.758, 1.074, 1.482, 0.995, 0.77, 1.535],
        [0.043, -0.751, 1.072, 1.484, 1.003, 0.766, 1.519],
        [0.051, -0.743, 1.069, 1.485, 1.011, 0.762, 1.503],
        [0.059, -0.736, 1.067, 1.487, 1.018, 0.758, 1.487],
        [0.067, -0.728, 1.065, 1.488, 1.026, 0.754, 1.471],
        [0.075, -0.721, 1.062, 1.49, 1.034, 0.75, 1.455],
        [0.083, -0.714, 1.06, 1.491, 1.041, 0.746, 1.439],
        [0.091, -0.706, 1.058, 1.493, 1.049, 0.742, 1.422],
        [0.099, -0.698, 1.055, 1.494, 1.057, 0.738, 1.406],
        [0.107, -0.691, 1.053, 1.496, 1.066, 0.734, 1.389],
        [0.115, -0.683, 1.05, 1.497, 1.074, 0.73, 1.373],
        [0.123, -0.675, 1.048, 1.499, 1.082, 0.725, 1.356],
        [0.131, -0.667, 1.045, 1.501, 1.091, 0.721, 1.339],
        [0.139, -0.659, 1.043, 1.502, 1.099, 0.717, 1.322],
        [0.147, -0.652, 1.041, 1.504, 1.108, 0.713, 1.305],
        [0.155, -0.644, 1.038, 1.506, 1.117, 0.709, 1.288],
        [0.163, -0.636, 1.036, 1.508, 1.125, 0.705, 1.271],
        [0.171, -0.628, 1.033, 1.509, 1.134, 0.7, 1.254],
        [0.179, -0.62, 1.031, 1.511, 1.142, 0.696, 1.237],
        [0.187, -0.612, 1.029, 1.512, 1.15, 0.692, 1.221],
        [0.195, -0.604, 1.027, 1.514, 1.158, 0.688, 1.204],
        [0.203, -0.597, 1.024, 1.516, 1.166, 0.685, 1.188],
        [0.21, -0.589, 1.022, 1.517, 1.173, 0.681, 1.172],
        [0.218, -0.582, 1.02, 1.518, 1.18, 0.677, 1.156],
        [0.225, -0.575, 1.018, 1.52, 1.187, 0.673, 1.14],
        [0.232, -0.568, 1.017, 1.521, 1.193, 0.67, 1.125],
        [0.24, -0.561, 1.015, 1.522, 1.199, 0.666, 1.111],
        [0.247, -0.555, 1.013, 1.523, 1.204, 0.663, 1.096],
        [0.254, -0.548, 1.012, 1.524, 1.209, 0.66, 1.082],
        [0.26, -0.542, 1.01, 1.524, 1.213, 0.657, 1.069],
        [0.267, -0.537, 1.009, 1.525, 1.217, 0.654, 1.056],
        [0.273, -0.531, 1.008, 1.525, 1.221, 0.651, 1.044],
        [0.279, -0.526, 1.007, 1.526, 1.224, 0.649, 1.033],
        [0.285, -0.521, 1.006, 1.526, 1.226, 0.646, 1.022],
        [0.29, -0.517, 1.005, 1.526, 1.229, 0.644, 1.012],
        [0.295, -0.513, 1.004, 1.527, 1.231, 0.642, 1.003],
        [0.299, -0.509, 1.003, 1.527, 1.233, 0.64, 0.994],
        [0.303, -0.506, 1.003, 1.527, 1.234, 0.639, 0.987],
        [0.307, -0.503, 1.002, 1.527, 1.235, 0.638, 0.981],
        [0.31, -0.501, 1.002, 1.527, 1.236, 0.636, 0.975],
        [0.312, -0.499, 1.002, 1.527, 1.237, 0.635, 0.971],
        [0.314, -0.498, 1.001, 1.527, 1.237, 0.635, 0.968],
        [0.315, -0.497, 1.001, 1.527, 1.237, 0.634, 0.966],
        [0.315, -0.497, 1.001, 1.527, 1.238, 0.634, 0.966]]
    joint_vel_goal_list = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.033, 0.038, -0.012, 0.009, 0.049, -0.019, -0.074],
                           [0.064, 0.073, -0.022, 0.017, 0.093, -0.036, -0.144],
                           [0.092, 0.106, -0.032, 0.024, 0.135, -0.053, -0.208],
                           [0.119, 0.136, -0.041, 0.031, 0.172, -0.068, -0.268],
                           [0.144, 0.163, -0.05, 0.037, 0.205, -0.081, -0.323],
                           [0.166, 0.188, -0.057, 0.042, 0.235, -0.094, -0.373],
                           [0.187, 0.211, -0.064, 0.047, 0.261, -0.105, -0.419],
                           [0.206, 0.23, -0.07, 0.051, 0.284, -0.116, -0.459],
                           [0.222, 0.248, -0.075, 0.055, 0.302, -0.124, -0.495],
                           [0.237, 0.262, -0.08, 0.058, 0.317, -0.132, -0.527],
                           [0.249, 0.274, -0.084, 0.06, 0.328, -0.138, -0.553],
                           [0.26, 0.283, -0.086, 0.062, 0.336, -0.144, -0.574],
                           [0.268, 0.29, -0.088, 0.063, 0.339, -0.148, -0.591],
                           [0.275, 0.294, -0.09, 0.063, 0.339, -0.15, -0.603],
                           [0.279, 0.296, -0.09, 0.063, 0.335, -0.152, -0.611],
                           [0.282, 0.295, -0.09, 0.062, 0.328, -0.152, -0.613],
                           [0.282, 0.291, -0.089, 0.06, 0.317, -0.151, -0.611],
                           [0.282, 0.288, -0.088, 0.059, 0.305, -0.15, -0.608],
                           [0.282, 0.284, -0.087, 0.057, 0.295, -0.149, -0.605],
                           [0.282, 0.281, -0.086, 0.056, 0.286, -0.148, -0.603],
                           [0.282, 0.278, -0.085, 0.055, 0.277, -0.147, -0.6],
                           [0.282, 0.275, -0.084, 0.054, 0.27, -0.146, -0.598],
                           [0.282, 0.273, -0.083, 0.053, 0.263, -0.145, -0.595],
                           [0.282, 0.271, -0.082, 0.052, 0.258, -0.144, -0.593],
                           [0.282, 0.269, -0.082, 0.051, 0.253, -0.144, -0.591],
                           [0.282, 0.267, -0.081, 0.05, 0.249, -0.143, -0.589],
                           [0.282, 0.266, -0.081, 0.05, 0.246, -0.142, -0.588],
                           [0.282, 0.265, -0.081, 0.049, 0.245, -0.142, -0.586],
                           [0.282, 0.265, -0.08, 0.049, 0.244, -0.141, -0.585],
                           [0.282, 0.264, -0.08, 0.048, 0.243, -0.141, -0.583],
                           [0.282, 0.264, -0.08, 0.048, 0.244, -0.14, -0.582],
                           [0.282, 0.264, -0.08, 0.048, 0.246, -0.14, -0.581],
                           [0.282, 0.265, -0.08, 0.048, 0.249, -0.14, -0.58],
                           [0.282, 0.266, -0.081, 0.049, 0.252, -0.139, -0.579],
                           [0.282, 0.266, -0.081, 0.049, 0.255, -0.139, -0.579],
                           [0.282, 0.267, -0.081, 0.049, 0.258, -0.139, -0.577],
                           [0.282, 0.267, -0.081, 0.049, 0.26, -0.139, -0.576],
                           [0.281, 0.267, -0.081, 0.049, 0.262, -0.138, -0.575],
                           [0.281, 0.267, -0.081, 0.049, 0.264, -0.138, -0.573],
                           [0.281, 0.266, -0.081, 0.049, 0.265, -0.138, -0.571],
                           [0.28, 0.266, -0.081, 0.049, 0.266, -0.138, -0.569],
                           [0.28, 0.265, -0.081, 0.049, 0.267, -0.137, -0.567],
                           [0.279, 0.264, -0.081, 0.049, 0.267, -0.137, -0.565],
                           [0.279, 0.263, -0.081, 0.048, 0.267, -0.137, -0.563],
                           [0.278, 0.262, -0.081, 0.048, 0.266, -0.136, -0.56],
                           [0.277, 0.26, -0.08, 0.048, 0.265, -0.136, -0.557],
                           [0.276, 0.259, -0.08, 0.048, 0.264, -0.136, -0.554],
                           [0.275, 0.257, -0.08, 0.047, 0.262, -0.135, -0.551],
                           [0.275, 0.255, -0.08, 0.047, 0.26, -0.135, -0.548],
                           [0.274, 0.253, -0.079, 0.047, 0.258, -0.135, -0.545],
                           [0.273, 0.251, -0.079, 0.046, 0.256, -0.134, -0.541],
                           [0.272, 0.249, -0.079, 0.046, 0.254, -0.134, -0.539],
                           [0.271, 0.248, -0.079, 0.046, 0.253, -0.134, -0.537],
                           [0.27, 0.246, -0.078, 0.046, 0.252, -0.134, -0.535],
                           [0.27, 0.246, -0.078, 0.046, 0.251, -0.134, -0.534],
                           [0.269, 0.245, -0.078, 0.046, 0.251, -0.134, -0.533],
                           [0.269, 0.245, -0.078, 0.047, 0.251, -0.134, -0.533],
                           [0.268, 0.245, -0.078, 0.047, 0.252, -0.134, -0.533],
                           [0.268, 0.245, -0.078, 0.047, 0.253, -0.134, -0.534],
                           [0.268, 0.246, -0.079, 0.048, 0.255, -0.135, -0.535],
                           [0.268, 0.247, -0.079, 0.049, 0.257, -0.135, -0.537],
                           [0.268, 0.248, -0.079, 0.05, 0.26, -0.135, -0.539],
                           [0.268, 0.25, -0.079, 0.05, 0.263, -0.136, -0.542],
                           [0.268, 0.252, -0.08, 0.051, 0.266, -0.137, -0.546],
                           [0.268, 0.254, -0.08, 0.053, 0.27, -0.137, -0.549],
                           [0.269, 0.257, -0.081, 0.054, 0.275, -0.138, -0.554],
                           [0.269, 0.259, -0.081, 0.055, 0.279, -0.139, -0.559],
                           [0.269, 0.262, -0.082, 0.056, 0.283, -0.139, -0.563],
                           [0.269, 0.263, -0.082, 0.057, 0.285, -0.139, -0.565],
                           [0.269, 0.264, -0.081, 0.057, 0.287, -0.139, -0.567],
                           [0.268, 0.265, -0.081, 0.057, 0.287, -0.139, -0.568],
                           [0.267, 0.265, -0.08, 0.057, 0.285, -0.138, -0.567],
                           [0.266, 0.264, -0.079, 0.056, 0.283, -0.137, -0.565],
                           [0.264, 0.263, -0.078, 0.056, 0.279, -0.136, -0.562],
                           [0.263, 0.261, -0.077, 0.054, 0.273, -0.135, -0.558],
                           [0.261, 0.258, -0.075, 0.053, 0.267, -0.133, -0.553],
                           [0.258, 0.255, -0.073, 0.051, 0.259, -0.131, -0.547],
                           [0.256, 0.251, -0.071, 0.049, 0.25, -0.128, -0.539],
                           [0.253, 0.246, -0.068, 0.046, 0.24, -0.126, -0.53],
                           [0.25, 0.241, -0.065, 0.043, 0.228, -0.123, -0.521],
                           [0.246, 0.235, -0.062, 0.04, 0.215, -0.119, -0.51],
                           [0.242, 0.229, -0.059, 0.037, 0.201, -0.116, -0.498],
                           [0.238, 0.221, -0.056, 0.033, 0.186, -0.112, -0.484],
                           [0.234, 0.214, -0.052, 0.029, 0.169, -0.108, -0.47],
                           [0.228, 0.205, -0.048, 0.025, 0.154, -0.103, -0.454],
                           [0.222, 0.196, -0.045, 0.022, 0.139, -0.098, -0.436],
                           [0.213, 0.187, -0.041, 0.019, 0.124, -0.093, -0.416],
                           [0.204, 0.176, -0.038, 0.016, 0.111, -0.088, -0.394],
                           [0.194, 0.165, -0.034, 0.013, 0.098, -0.082, -0.371],
                           [0.182, 0.153, -0.031, 0.011, 0.085, -0.076, -0.346],
                           [0.169, 0.141, -0.028, 0.008, 0.074, -0.07, -0.319],
                           [0.155, 0.128, -0.024, 0.006, 0.063, -0.063, -0.291],
                           [0.14, 0.114, -0.021, 0.005, 0.053, -0.056, -0.261],
                           [0.124, 0.1, -0.018, 0.003, 0.043, -0.049, -0.229],
                           [0.106, 0.085, -0.015, 0.002, 0.034, -0.042, -0.195],
                           [0.087, 0.069, -0.012, 0.001, 0.026, -0.034, -0.159],
                           [0.067, 0.053, -0.009, 0.0, 0.018, -0.026, -0.122],
                           [0.046, 0.036, -0.006, 0.0, 0.012, -0.018, -0.083],
                           [0.024, 0.018, -0.003, 0.0, 0.005, -0.009, -0.042],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    """ 初始化变量 """
    point_sum = len(joint_angles_goal_list)
    point_now = 0
    joint_angles_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_velocities_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 速度误差
    z2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dy_tau = right_arm.joint_efforts()  # 实际关节转矩

    count = 2500  # 迭代次数
    output_size = 100
    out_ratio = count / output_size  # 作图抽取率

    """ 画图用初始化，输入关节力，关节实际位置，关节期望位置，时间，末端实际位置，末端期望位置 """
    joint_effort_display = np.zeros((7, output_size + 1), dtype=float)
    joint_actual_pos_display = np.zeros((7, output_size + 1), dtype=float)
    joint_req_pos_display = np.zeros((7, output_size + 1), dtype=float)
    tout = np.zeros((7, output_size + 1), dtype=float)
    xyz_display = np.zeros((3, output_size + 1), dtype=float)
    xyz_req_display = np.zeros((3, output_size + 1), dtype=float)
    z3_display = np.zeros((7, output_size + 1), dtype=float)
    sat_display = np.zeros((1, output_size + 1), dtype=float)

    """ 产生高斯白噪声 """
    mu = 0
    sigma = 0.3
    f_dis = random.gauss(mu, sigma)

    a = 0  # 输出用计数

    for i in range(0, count):
        if not rospy.is_shutdown():  # 循环过程中ctrl+c判断
            """ 得到角度,把字典形式写成序列形式 """
            joint_angles_now = right_arm.joint_angles()
            temp = 0
            for key in joint_angles_now:
                joint_angles_now_list[temp] = joint_angles_now[key]
                temp = temp + 1

            """ 得到速度,把字典形式写成序列形式 """
            joint_velocities_now = right_arm.joint_velocities()
            temp = 0
            for key in joint_velocities_now:
                joint_velocities_now_list[temp] = joint_velocities_now[key]
                temp = temp + 1

            """ 计算出当前应该的目标点 """
            if point_now < point_sum:  # point_sum等于100，每8个迭代点算一次目标值
                if i % 8 == 0:
                    point_now = point_now + 1

            """ 计算z1,计算关节角误差，上面8个+1，因此这里计算当前-1 """
            for aaa in range(0, 7):
                z1[aaa] = joint_angles_now_list[aaa] - joint_angles_goal_list[point_now - 1][aaa]

            """ 计算alpha """
            for aaa in range(0, 7):
                alpha[aaa] = -K1_R[aaa] * z1[aaa] + joint_vel_goal_list[point_now - 1][aaa]

            """ 计算z2 """
            for aaa in range(0, 7):
                z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]

            """ 得到通信计算的值，即神经网络部分的值,udp_l.controller.torController[temp].get_kd()只是7个数 """
            temp = 0
            for key in dy_tau:
                dy_tau[key] = -z1[temp] - K2_R[aaa] * z2[temp]
                temp = temp + 1

            """ 画图初始化 """
            sat = 2.0
            if a == 0:
                temp = 0
                xyz_pose = right_arm.endpoint_pose()
                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.4
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4

                start_time = rospy.get_time()
                get_pose = right_arm.joint_angles()
                z3_display[temp, a] = 0
                sat_display[0, a] = sat
                # 把字典转换成序列
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]

                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now - 1][temp]
                    tout[temp, a] = 0
                    z3_display[temp, a] = 0
                    temp = temp + 1
                a = a + 1

            right_arm.set_joint_torques(dy_tau)

            """ 作图用 """
            if i % out_ratio == 0:
                display_cur_time = rospy.get_time() # 控制后的时间
                """ 关节角度 """
                temp = 0
                get_pose = right_arm.joint_angles()
                xyz_pose = right_arm.endpoint_pose()

                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.4 + 0.004 * a
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4
                sat_display[0, a] = sat
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    if joint_effort_display[temp, a] > sat:
                        joint_effort_display[temp, a] = sat
                    if joint_effort_display[temp, a] < -1 * sat:
                        joint_effort_display[temp, a] = -1 * sat

                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now - 1][temp]
                    tout[temp, a] = display_cur_time - start_time

                    """ 计算扰动 """
                    f_dis = 2 * random.gauss(mu, sigma)
                    deta_tau = joint_effort_display[temp, a] - dy_tau[key]
                    K = 2
                    D = -f_dis + deta_tau
                    z3_display[temp, a] = D - K * z2[temp]

                    temp = temp + 1

                a = a + 1
            Rate.sleep()
    right_arm.exit_control_mode()

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    plt.title("Right joint S0", font1)
    lns1 = ax.plot(tout[0].T, joint_actual_pos_display[0], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[0].T, joint_req_pos_display[0], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[0] - joint_actual_pos_display[0], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig1.tight_layout()
    plt.savefig('picture/right/right_joint_s0.eps', format='eps')

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    plt.title("Right joint S1", font1)
    lns1 = ax.plot(tout[1].T, joint_actual_pos_display[1], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[1].T, joint_req_pos_display[1], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[1] - joint_actual_pos_display[1], '-*', linewidth=3, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig2.tight_layout()
    plt.savefig('picture/right/right_joint_s1.eps', format='eps')

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    plt.title("Right joint W0", font1)
    lns1 = ax.plot(tout[2].T, joint_actual_pos_display[2], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[2].T, joint_req_pos_display[2], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[2] - joint_actual_pos_display[2], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig3.tight_layout()
    plt.savefig('picture/right/right_joint_w0.eps', format='eps')

    fig4 = plt.figure(4)
    ax = fig4.add_subplot(111)
    plt.title("Right joint W1", font1)
    lns1 = ax.plot(tout[3].T, joint_actual_pos_display[3], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[3].T, joint_req_pos_display[3], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[3] - joint_actual_pos_display[3], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig4.tight_layout()
    plt.savefig('picture/right/right_joint_w1.eps', format='eps')

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    plt.title("Right joint W2", font1)
    lns1 = ax.plot(tout[4].T, joint_actual_pos_display[4], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[4].T, joint_req_pos_display[4], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[4] - joint_actual_pos_display[4], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig5.tight_layout()
    plt.savefig('picture/right/right_joint_w2.eps', format='eps')

    fig6 = plt.figure(6)
    ax = fig6.add_subplot(111)
    plt.title("Right joint E0", font1)
    lns1 = ax.plot(tout[5].T, joint_actual_pos_display[5], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[5].T, joint_req_pos_display[5], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[5] - joint_actual_pos_display[5], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig6.tight_layout()
    plt.savefig('picture/right/right_joint_E0.eps', format='eps')

    fig7 = plt.figure(7)
    ax = fig7.add_subplot(111)
    plt.title("Right joint E1", font1)
    lns1 = ax.plot(tout[6].T, joint_actual_pos_display[6], '-', linewidth=linesize, color="red", label="Actual value")
    lns2 = ax.plot(tout[6].T, joint_req_pos_display[6], '-o', linewidth=linesize, color="green", label="Desired value")
    ax2 = ax.twinx()
    lns3 = ax2.plot(tout[0].T, joint_req_pos_display[6] - joint_actual_pos_display[6], '-*', linewidth=linesize, color="blue",
                    label="Error value")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize=legend_size)
    ax.grid()
    ax.set_xlabel("Time/s", font2)
    ax.set_ylabel("Angle/rad", font2)
    ax2.set_ylabel("Angle error/rad", font2)
    ax.set_xlim(0, 7.3)
    ax.tick_params(labelsize=size2)
    ax2.tick_params(labelsize=size2)
    fig7.tight_layout()
    plt.savefig('picture/right/right_joint_E1.eps', format='eps')

    fig8 = plt.figure(8)
    ax = fig8.add_subplot(111)
    plt.title("Right torques")
    plt.plot(tout[0].T, joint_effort_display[0], '-*', linewidth=linesize, label='Joint S0')
    plt.plot(tout[0].T, joint_effort_display[1], '-o', linewidth=linesize, label='Joint S1')
    plt.plot(tout[0].T, joint_effort_display[2], '-x', linewidth=linesize, label='Joint W0')
    plt.plot(tout[0].T, joint_effort_display[3], '-s', linewidth=linesize, label='Joint W1')
    plt.plot(tout[0].T, joint_effort_display[4], '-p', linewidth=linesize, label='Joint W2')
    plt.plot(tout[0].T, joint_effort_display[5], '-h', linewidth=linesize, label='Joint E0')
    plt.plot(tout[0].T, joint_effort_display[6], '-d', linewidth=linesize, label='Joint E1')
    plt.plot(tout[0].T, sat_display[0], 'm--', linewidth=linesize, label='Limit value')
    plt.plot(tout[0].T, -1*sat_display[0], 'm--', linewidth=linesize)
    plt.xlabel("Time/s", font2)
    plt.ylabel("Torque/Nm", font2)
    ax.grid()
    plt.xlim(0, 7.3)
    plt.ylim(-2.5, 2.5)
    # plt.legend(bbox_to_anchor=(1.105, 0.6), fontsize=legend_size)
    plt.legend(loc=7, fontsize=legend_size)
    fig8.tight_layout()
    plt.savefig('picture/right/right_torques.eps', format='eps')

    fig9 = plt.figure(9)
    ax = fig9.add_subplot(111, projection='3d')
    ax.grid()
    plt.title("Right endpoint position")
    ax.plot(xyz_display[0], xyz_display[1], xyz_display[2], '-o', label="Actual value", color='red', linewidth=linesize)
    ax.plot(xyz_req_display[0], xyz_req_display[1], xyz_req_display[2], '-*', label="Desired value", color='green', linewidth=linesize)
    ax.set_xlabel("X/m", font2)
    ax.set_ylabel("Y/m", font2)
    ax.set_zlabel("Z/m", font2)
    plt.ylim(-1, 1)
    ax.set_zlim(0.2, 0.5)
    # ax.set_xlim(0, 1)
    fig9.tight_layout()
    # plt.legend(loc='best', bbox_to_anchor=(1.1, 1), fontsize=legend_size)
    plt.legend(loc=0, fontsize=legend_size)
    plt.savefig('picture/right/right_endpoint_position.eps', format='eps')

    fig10 = plt.figure(10)
    ax = fig10.add_subplot(111)
    ax.grid()
    plt.title("Right z3")
    plt.plot(tout[0].T, z3_display[0], '-*', linewidth=linesize, label='Joint S0')
    plt.plot(tout[0].T, z3_display[1], '-o', linewidth=linesize, label='Joint S1')
    plt.plot(tout[0].T, z3_display[2], '-x', linewidth=linesize, label='Joint W0')
    plt.plot(tout[0].T, z3_display[3], '-s', linewidth=linesize, label='Joint W1')
    plt.plot(tout[0].T, z3_display[4], '-p', linewidth=linesize, label='Joint W2')
    plt.plot(tout[0].T, z3_display[5], '-h', linewidth=linesize, label='Joint E0')
    plt.plot(tout[0].T, z3_display[6], '-d', linewidth=linesize, label='Joint E1')
    plt.xlabel("Time/s", font2)
    plt.ylabel("Torque/Nm", font2)
    plt.xlim(0, 7.3)
    # plt.ylim(-4, 4)
    # plt.legend(bbox_to_anchor=(1.095, 0.6), fontsize=legend_size)
    plt.legend(loc=7, fontsize=legend_size)
    fig10.tight_layout()
    plt.savefig('picture/right/right_z3.eps', format='eps')

    # plt.show()

def move_left_init():
    left_arm = baxter_interface.Limb("left")

    t1 = left_arm.joint_angles()
    t2 = [-1.05, 1.245, -0.677, -1.134, 2.358, 0.472, -0.935]
    temp = 0
    for key in t1:
        t1[key] = t2[temp]
        temp = temp + 1
    left_arm.move_to_joint_positions(t1)

def move_right_init():
    right_arm = baxter_interface.Limb("right")

    t3 = right_arm.joint_angles()
    t4 = [-0.397, -1.181, 1.204, 1.4, 0.556, 0.991, 2.436]
    temp = 0
    for key in t3:
        t3[key] = t4[temp]
        temp = temp + 1
    right_arm.move_to_joint_positions(t3)

def main():
    """ 初始化机器人节点和机器人 """
    rospy.init_node("dual_arm_NN")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    rs.enable()

    try:
        thread_l_init = threading.Thread(target=move_left_init)
        thread_r_init = threading.Thread(target=move_right_init)
        thread_l_init.start()
        thread_r_init.start()
        thread_l_init.join()
        thread_r_init.join()
    except rospy.ROSInterruptException:
        pass

    try:
        left_arm_control()
        # thread_l = threading.Thread(target=left_arm_control)
        # thread_r = threading.Thread(target=right_arm_control)
        # thread_l.start()
        # thread_r.start()
        # thread_l.join()
        # thread_r.join()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    # K1,K2 w0 w1 w2 e0 e1 s0 s1
    K1_L = np.array([15.7, 22, 20.3, 12.6, 15.0, 17.7, 20.0])
    K2_L = np.array([1.2, 2.0, 5.1, 10.1, 4.5, 2.1, 2.2])
    # K1 = np.array([20.6, 20.0, 22.0, 12.6, 17.7, 17.7, 15.7])
    # K2 = np.array([12.1, 8.5, 4.0, 5.1, 3.1, 2.8, 4.2])

    # K1_R,K2_R       s0 s1 w0 w1 w2 e0 e1
    # K1_R = np.array([14.6, 15.0, 17.7, 15.0, 15.7, 10.02, 10.3])
    # K2_R = np.array([2.1, 4.5, 5.1, 18.0, 1.2, 2.5, 2.1])
    K1_R = np.array([17.7, 15, 18.0, 22.0, 16.7, 26.0, 10.3])
    K2_R = np.array([5.1, 18, 4.1, 4.5, 4.2, 3.5, 2.1])
    main()
