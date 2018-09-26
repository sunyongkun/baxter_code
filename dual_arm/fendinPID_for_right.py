#!/usr/bin/env python
# coding=utf-8
import math
import numpy as np
import baxter_interface
import sys
import rospy
from moveit_commander import conversions
import matplotlib.pyplot as plt
import threading
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
#from baxter_pykdl import baxter_kinematics
from baxter_core_msgs.msg import SEAJointState
from baxter_core_msgs.msg import JointCommand
from tf import transformations
# from beginner_tutorials.srv import dynamics
# from beginner_tutorials.srv import dynamicsRequest
from mpl_toolkits.mplot3d import Axes3D

import struct
import socket


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

class PIDController(object):
    def __init__(self):
        self._prev_err = 0.0

        self._kp = 0.0
        self._ki = 0.0
        self._kd = 0.0
        # initialize error, results
        self._cp = 0.0
        self._ci = 0.0
        self._cd = 0.0

        self._cur_time = 0.0
        self._prev_time = 0.0

        self.initialize()

    def initialize(self):
        self._cur_time = rospy.get_time()
        self._prev_time = self._cur_time

        self._prev_err = 0.0

        self._cp = 0.0
        self._ci = 0.0
        self._cd = 0.0

    def set_kp(self, invar):
        self._kp = invar

    def set_ki(self, invar):
        self._ki = invar

    def set_kd(self, invar):
        self._kd = invar

    def get_kp(self):
        return self._kp

    def get_ki(self):
        return self._ki

    def get_kd(self):
        return self._kd

    def compute_output(self, error):
        """
        Performs a PID computation and returns a control value based on
        the elapsed time (dt) and the error signal from a summing junction
        (the error parameter).
        """
        self._cur_time = rospy.get_time()  # get t
        dt = self._cur_time - self._prev_time  # get delta t
        de = error - self._prev_err  # get delta error
        if error <=0.01 and error>=-0.01:
            self._cp = 0  # proportional term
        else:
            self._cp = error
        self._ci += error * dt  # integral term

        self._cd = 0
        if dt > 0:  # no div by zero
            self._cd = de / dt  # derivative term

        self._prev_time = self._cur_time  # save t for next pass
        self._prev_err = error  # save t-1 error

        result = ((self._kp * self._cp) + (self._ki * self._ci) + (self._kd * self._cd))
        return  result

class JointControl(object):
    def __init__(self, ArmName):
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        self.torcmd_now = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        self.name = ArmName
        self.limb = baxter_interface.Limb(ArmName)
        #self.limb.move_to_neutral()

        t1 = self.limb.joint_angles()
        t2 = [-0.397, -1.181, 1.204, 1.4, 0.556, 0.991, 2.436]
        temp = 0
        for key in t1:
            t1[key] = t2[temp]
            temp = temp + 1
        self.limb.move_to_joint_positions(t1)
        #self.limb.set_joint_position_speed(0.1)

        self.actual_effort = self.limb.joint_efforts()
        self.gravity_torques = self.limb.joint_efforts()
        self.final_torques = self.gravity_torques.copy()

        self.qnow = dict()  # 得到关节角度的字典
        self.qnow_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 得到角度的值
        self.torcmd = dict()  # 给baxter赋值

        self.torController0 = PIDController()
        self.torController1 = PIDController()
        self.torController2 = PIDController()
        self.torController3 = PIDController()
        self.torController4 = PIDController()
        self.torController5 = PIDController()
        self.torController6 = PIDController()

        self.torController = \
            [self.torController0, self.torController1, self.torController2, self.torController3,
             self.torController4, self.torController5, self.torController6
             ]

        '''设置PID参数'''
        # 最前端

        # s0
        self.torController[0].set_kp(17.7)  # 130#80.0#*0.6
        self.torController[0].set_ki(0.01)
        self.torController[0].set_kd(3.1)  # 10#15#0.01#*0.6#21.0

        # s1
        self.torController[1].set_kp(15)  # 130#80.0#*0.6
        self.torController[1].set_ki(6)
        self.torController[1].set_kd(3)  # 10#15#0.01#*0.6#21.0

        # w0
        self.torController[2].set_kp(18.7)  # 15.7
        self.torController[2].set_ki(1)
        self.torController[2].set_kd(5.2)  # 1.2

        # w1
        self.torController[3].set_kp(26.02)  # 10  12
        self.torController[3].set_ki(1.2)
        self.torController[3].set_kd(2.5)  # 2.5

        # w2
        self.torController[4].set_kp(10.3)
        self.torController[4].set_ki(0.1)  # 0.1
        self.torController[4].set_kd(2.1)

        # e0
        self.torController[5].set_kp(18)  # 14.6
        self.torController[5].set_ki(1.5)  # 0.05
        self.torController[5].set_kd(3.1)  # 10#15#0.01#*0.6#21.0

        # e1
        self.torController[6].set_kp(20)  # 130#80.0#*0.6
        self.torController[6].set_ki(1.5)
        self.torController[6].set_kd(3.5)  # 10#15#0.01#*0.6#21.0

        # self.subscribe_to_gravity_compensation()

    '''力矩控制'''
    def torquecommand(self, qd):  # qd为期望轨迹
        self.err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.qnow = self.limb.joint_angles()  # 得到每个关节的当前角度

        temp = 0
        for key in self.qnow:
            self.qnow_value[temp] = self.qnow[key]  # 转化为list
            temp = temp + 1
        # print self.qnow_value
        self.err = qd - self.qnow_value  # 计算每个关节角度的误差
        # print self.err
        self.torcmd = self.limb.joint_efforts()
        # print self.torcmd
        temp = 0
        for key in self.torcmd:
            self.torcmd[key] = self.torController[temp].compute_output(self.err[temp])
            temp = temp + 1


        self.final_torques = self.gravity_torques.copy()  ##########又把final 和gravity连在一起了######
        for key in self.torcmd:
            self.final_torques[key] = self.torcmd[key]
        return self.torcmd

    def gravity_callback(self, data):
        frommsg1 = data.gravity_model_effort
        frommsg2 = data.actual_effort

        temp = 0
        for key in self.gravity_torques:
            self.gravity_torques[key] = frommsg1[temp]
            self.actual_effort[key] = frommsg2[temp]
            temp = temp + 1
            # self.limb.set_joint_torques(self.gravity_torques)

    def get_ik_solution(self, rpy_pose):
        quaternion_pose = conversions.list_to_pose_stamped(rpy_pose, "base")
        node = "ExternalTools/" + "right" + "/PositionKinematicsNode/IKService"
        ik_service = rospy.ServiceProxy(node, SolvePositionIK)
        ik_request = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id="base")
        ik_request.pose_stamp.append(quaternion_pose)
        try:
            rospy.wait_for_service(node, 15.0)  # 5改成了15
            ik_response = ik_service(ik_request)
        except (rospy.ServiceException, rospy.ROSException), error_message:
            rospy.logerr("Service request failed: %r" % (error_message,))
            sys.exit("ERROR - baxter_ik_move - Failed to append pose")

        if ik_response.isValid[0]:
            print("PASS: Valid joint configuration found")
            # convert response to joint position control dictionary
            limb_joints = dict(zip(ik_response.joints[0].name, ik_response.joints[0].position))
            return limb_joints

        else:
            sys.exit("ERROR - baxter_ik_move - No valid joint configuration found")

    def subscribe_to_gravity_compensation(self):
        topic_str = "robot/limb/right/gravity_compensation_torques"
        rospy.Subscriber(topic_str, SEAJointState, self.gravity_callback)

    '''改变控制模式，还不清楚消息怎么发送
    def publish_to_changeto_positon_mode(self):
        topic_str = "/robot/limb/"+"right"+"/joint_command"
        pub = rospy.Publisher(topic_str,JointCommand,queue_size=10)
        pub.publish(,)
    '''

    def shutdown_close(self):
        #self.limb.move_to_neutral()
        print("shutdown.........")

'''最大的类'''
class from_udp(object):
    def __init__(self, ArmName):
        self.controller = JointControl(ArmName)
        self.limb = baxter_interface.Limb(ArmName)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP通信
        self.s.bind(('10.1.1.20', 6666)) #绑定本机的地址，端口号识别程序


        self.angles = self.limb.joint_angles()
        print self.angles
        self.trans_angles_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_trans_angles_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.velocities = self.limb.joint_velocities()
        self.trans_velocities_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_trans_velocities_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.trans_z2_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_z2_alphas_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.pre_alphas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cur_alphas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.sub = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        self._pre_time = rospy.get_time()
        self._cur_time = self._pre_time

        self.Rate = rospy.Rate(500)
        '''线程'''
        self.thread_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.real_thread_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.thread_stop = False
        self.thread = threading.Thread(target=self.commucate)
        self.start = 1
        self.thread.start()

    def commucate(self):
        while not self.thread_stop:
            self._cur_time = rospy.get_time()
            dt = self._cur_time - self._pre_time

            '''需要传递的角度'''
            self.angles = self.limb.joint_angles()
            temp = 0
            for key in self.angles:
                self.trans_angles_list[temp] = self.angles[key]
                temp = temp + 1

            self.real_trans_angles_list[0] = self.trans_angles_list[0]
            self.real_trans_angles_list[1] = self.trans_angles_list[1]
            self.real_trans_angles_list[2] = self.trans_angles_list[5]
            self.real_trans_angles_list[3] = self.trans_angles_list[6]
            self.real_trans_angles_list[4] = self.trans_angles_list[2]
            self.real_trans_angles_list[5] = self.trans_angles_list[3]
            self.real_trans_angles_list[6] = self.trans_angles_list[4]


            # print self.real_trans_angles_list
            for i in range(0, 7):
                self.real_trans_angles_list[i] = self.real_trans_angles_list[i] * 1000.0 + 32768.0

            '''需要传递的速度'''
            self.velocities = self.limb.joint_velocities()
            temp = 0
            for key in self.velocities:
                self.trans_velocities_list[temp] = self.velocities[key]
                temp = temp + 1

            self.real_trans_velocities_list[0] = self.trans_velocities_list[0]
            self.real_trans_velocities_list[1] = self.trans_velocities_list[1]
            self.real_trans_velocities_list[2] = self.trans_velocities_list[5]
            self.real_trans_velocities_list[3] = self.trans_velocities_list[6]
            self.real_trans_velocities_list[4] = self.trans_velocities_list[2]
            self.real_trans_velocities_list[5] = self.trans_velocities_list[3]
            self.real_trans_velocities_list[6] = self.trans_velocities_list[4]
            # print self.real_trans_velocities_list
            for i in range(0, 7):
                self.real_trans_velocities_list[i] = self.real_trans_velocities_list[i] * 1000.0 + 32768.0


            '''需要传递的z2'''

            self.real_z2_alphas_list[0] = self.trans_z2_list[0]
            self.real_z2_alphas_list[1] = self.trans_z2_list[1]
            self.real_z2_alphas_list[2] = self.trans_z2_list[5]
            self.real_z2_alphas_list[3] = self.trans_z2_list[6]
            self.real_z2_alphas_list[4] = self.trans_z2_list[2]
            self.real_z2_alphas_list[5] = self.trans_z2_list[3]
            self.real_z2_alphas_list[6] = self.trans_z2_list[4]


            for i in range(0, 7):
                self.real_z2_alphas_list[i] = self.real_z2_alphas_list[i] * 1000.0 + 32768.0

            self.msg = struct.pack("H", self.start)
            self.msg += struct.pack("7H", self.real_trans_angles_list[0], self.real_trans_angles_list[1], self.real_trans_angles_list[2], self.real_trans_angles_list[3], self.real_trans_angles_list[4], self.real_trans_angles_list[5], self.real_trans_angles_list[6])
            self.msg += struct.pack("7H", self.real_trans_velocities_list[0], self.real_trans_velocities_list[1], self.real_trans_velocities_list[2], self.real_trans_velocities_list[3], self.real_trans_velocities_list[4], self.real_trans_velocities_list[5], self.real_trans_velocities_list[6])
            self.msg += struct.pack("7H", self.real_z2_alphas_list[0], self.real_z2_alphas_list[1], self.real_z2_alphas_list[2], self.real_z2_alphas_list[3], self.real_z2_alphas_list[4], self.real_z2_alphas_list[5], self.real_z2_alphas_list[6])

            self.s.sendto(self.msg, ('10.1.1.21',8001))
            data,addr = self.s.recvfrom(1024)
            for i in range(0, 7):
                self.thread_result[i] = ((ord(data[2 * i]) * 256 + ord(data[2 * i + 1])) - 32768.0) / 1000.0

            self.real_thread_result[0] = self.thread_result[0]
            self.real_thread_result[1] = self.thread_result[1]
            self.real_thread_result[2] = self.thread_result[4]
            self.real_thread_result[3] = self.thread_result[5]
            self.real_thread_result[4] = self.thread_result[6]
            self.real_thread_result[5] = self.thread_result[2]
            self.real_thread_result[6] = self.thread_result[3]

            # print "self.real_thread_result"
            #print self.real_thread_result


            for i in range(0, 7):
                self.pre_alphas[i] = self.cur_alphas[i]
            self._pre_time = self._cur_time
            self.Rate.sleep()

def main():

    rospy.init_node("PID_controller_test")
    Rate = rospy.Rate(200)
    # 类实例化

    udp = from_udp('right')
    endpoint_pose_init = udp.controller.limb.endpoint_pose()
    endpoint_pose = endpoint_pose_init['position']

    pose_init = udp.controller.limb.joint_angles()
    print pose_init

    joint_goal_init = udp.controller.limb.joint_angles()
    joint_angles_goal_list = [[-0.397, -1.181, 1.204, 1.4, 0.556, 0.991, 2.436] ,
[-0.396, -1.18, 1.204, 1.4, 0.556, 0.991, 2.434] ,
[-0.395, -1.179, 1.203, 1.4, 0.559, 0.99, 2.431] ,
[-0.393, -1.176, 1.202, 1.401, 0.562, 0.989, 2.426] ,
[-0.389, -1.172, 1.201, 1.402, 0.567, 0.987, 2.419] ,
[-0.386, -1.168, 1.2, 1.403, 0.572, 0.985, 2.41] ,
[-0.381, -1.162, 1.198, 1.404, 0.579, 0.982, 2.399] ,
[-0.376, -1.156, 1.196, 1.405, 0.586, 0.979, 2.388] ,
[-0.37, -1.15, 1.194, 1.407, 0.595, 0.976, 2.374] ,
[-0.363, -1.143, 1.192, 1.408, 0.603, 0.972, 2.36] ,
[-0.356, -1.135, 1.19, 1.41, 0.613, 0.968, 2.345] ,
[-0.349, -1.127, 1.187, 1.412, 0.622, 0.964, 2.328] ,
[-0.341, -1.119, 1.185, 1.414, 0.632, 0.96, 2.312] ,
[-0.333, -1.11, 1.182, 1.416, 0.642, 0.956, 2.294] ,
[-0.325, -1.101, 1.18, 1.417, 0.653, 0.951, 2.276] ,
[-0.317, -1.092, 1.177, 1.419, 0.663, 0.947, 2.258] ,
[-0.309, -1.083, 1.174, 1.421, 0.673, 0.942, 2.24] ,
[-0.3, -1.075, 1.171, 1.423, 0.682, 0.938, 2.221] ,
[-0.292, -1.066, 1.169, 1.425, 0.692, 0.933, 2.203] ,
[-0.283, -1.057, 1.166, 1.427, 0.701, 0.929, 2.185] ,
[-0.275, -1.049, 1.164, 1.428, 0.709, 0.924, 2.167] ,
[-0.266, -1.041, 1.161, 1.43, 0.718, 0.92, 2.149] ,
[-0.258, -1.032, 1.159, 1.432, 0.726, 0.915, 2.131] ,
[-0.249, -1.024, 1.156, 1.433, 0.734, 0.911, 2.113] ,
[-0.241, -1.016, 1.154, 1.435, 0.742, 0.907, 2.095] ,
[-0.232, -1.008, 1.151, 1.436, 0.75, 0.902, 2.077] ,
[-0.224, -1.0, 1.149, 1.438, 0.757, 0.898, 2.059] ,
[-0.215, -0.992, 1.146, 1.439, 0.765, 0.894, 2.042] ,
[-0.207, -0.984, 1.144, 1.441, 0.772, 0.89, 2.024] ,
[-0.198, -0.976, 1.141, 1.442, 0.779, 0.885, 2.007] ,
[-0.19, -0.968, 1.139, 1.444, 0.787, 0.881, 1.989] ,
[-0.182, -0.96, 1.137, 1.445, 0.794, 0.877, 1.972] ,
[-0.173, -0.952, 1.134, 1.447, 0.801, 0.873, 1.954] ,
[-0.165, -0.944, 1.132, 1.448, 0.809, 0.868, 1.937] ,
[-0.156, -0.936, 1.129, 1.449, 0.816, 0.864, 1.919] ,
[-0.148, -0.928, 1.127, 1.451, 0.824, 0.86, 1.902] ,
[-0.139, -0.92, 1.124, 1.452, 0.831, 0.856, 1.885] ,
[-0.131, -0.912, 1.122, 1.454, 0.839, 0.852, 1.867] ,
[-0.122, -0.904, 1.12, 1.455, 0.847, 0.848, 1.85] ,
[-0.114, -0.896, 1.117, 1.457, 0.855, 0.843, 1.833] ,
[-0.105, -0.888, 1.115, 1.458, 0.863, 0.839, 1.816] ,
[-0.097, -0.88, 1.112, 1.46, 0.871, 0.835, 1.798] ,
[-0.089, -0.872, 1.11, 1.461, 0.879, 0.831, 1.781] ,
[-0.08, -0.864, 1.107, 1.463, 0.887, 0.827, 1.764] ,
[-0.072, -0.856, 1.105, 1.464, 0.895, 0.823, 1.748] ,
[-0.064, -0.849, 1.103, 1.466, 0.903, 0.819, 1.731] ,
[-0.055, -0.841, 1.1, 1.467, 0.911, 0.815, 1.714] ,
[-0.047, -0.833, 1.098, 1.468, 0.919, 0.811, 1.697] ,
[-0.039, -0.825, 1.095, 1.47, 0.927, 0.807, 1.681] ,
[-0.03, -0.818, 1.093, 1.471, 0.935, 0.802, 1.664] ,
[-0.022, -0.81, 1.091, 1.473, 0.942, 0.798, 1.648] ,
[-0.014, -0.802, 1.088, 1.474, 0.95, 0.794, 1.632] ,
[-0.006, -0.795, 1.086, 1.475, 0.958, 0.79, 1.615] ,
[0.002, -0.787, 1.084, 1.477, 0.965, 0.786, 1.599] ,
[0.01, -0.78, 1.081, 1.478, 0.973, 0.782, 1.583] ,
[0.019, -0.773, 1.079, 1.48, 0.98, 0.778, 1.567] ,
[0.027, -0.765, 1.076, 1.481, 0.988, 0.774, 1.551] ,
[0.035, -0.758, 1.074, 1.482, 0.995, 0.77, 1.535] ,
[0.043, -0.751, 1.072, 1.484, 1.003, 0.766, 1.519] ,
[0.051, -0.743, 1.069, 1.485, 1.011, 0.762, 1.503] ,
[0.059, -0.736, 1.067, 1.487, 1.018, 0.758, 1.487] ,
[0.067, -0.728, 1.065, 1.488, 1.026, 0.754, 1.471] ,
[0.075, -0.721, 1.062, 1.49, 1.034, 0.75, 1.455] ,
[0.083, -0.714, 1.06, 1.491, 1.041, 0.746, 1.439] ,
[0.091, -0.706, 1.058, 1.493, 1.049, 0.742, 1.422] ,
[0.099, -0.698, 1.055, 1.494, 1.057, 0.738, 1.406] ,
[0.107, -0.691, 1.053, 1.496, 1.066, 0.734, 1.389] ,
[0.115, -0.683, 1.05, 1.497, 1.074, 0.73, 1.373] ,
[0.123, -0.675, 1.048, 1.499, 1.082, 0.725, 1.356] ,
[0.131, -0.667, 1.045, 1.501, 1.091, 0.721, 1.339] ,
[0.139, -0.659, 1.043, 1.502, 1.099, 0.717, 1.322] ,
[0.147, -0.652, 1.041, 1.504, 1.108, 0.713, 1.305] ,
[0.155, -0.644, 1.038, 1.506, 1.117, 0.709, 1.288] ,
[0.163, -0.636, 1.036, 1.508, 1.125, 0.705, 1.271] ,
[0.171, -0.628, 1.033, 1.509, 1.134, 0.7, 1.254] ,
[0.179, -0.62, 1.031, 1.511, 1.142, 0.696, 1.237] ,
[0.187, -0.612, 1.029, 1.512, 1.15, 0.692, 1.221] ,
[0.195, -0.604, 1.027, 1.514, 1.158, 0.688, 1.204] ,
[0.203, -0.597, 1.024, 1.516, 1.166, 0.685, 1.188] ,
[0.21, -0.589, 1.022, 1.517, 1.173, 0.681, 1.172] ,
[0.218, -0.582, 1.02, 1.518, 1.18, 0.677, 1.156] ,
[0.225, -0.575, 1.018, 1.52, 1.187, 0.673, 1.14] ,
[0.232, -0.568, 1.017, 1.521, 1.193, 0.67, 1.125] ,
[0.24, -0.561, 1.015, 1.522, 1.199, 0.666, 1.111] ,
[0.247, -0.555, 1.013, 1.523, 1.204, 0.663, 1.096] ,
[0.254, -0.548, 1.012, 1.524, 1.209, 0.66, 1.082] ,
[0.26, -0.542, 1.01, 1.524, 1.213, 0.657, 1.069] ,
[0.267, -0.537, 1.009, 1.525, 1.217, 0.654, 1.056] ,
[0.273, -0.531, 1.008, 1.525, 1.221, 0.651, 1.044] ,
[0.279, -0.526, 1.007, 1.526, 1.224, 0.649, 1.033] ,
[0.285, -0.521, 1.006, 1.526, 1.226, 0.646, 1.022] ,
[0.29, -0.517, 1.005, 1.526, 1.229, 0.644, 1.012] ,
[0.295, -0.513, 1.004, 1.527, 1.231, 0.642, 1.003] ,
[0.299, -0.509, 1.003, 1.527, 1.233, 0.64, 0.994] ,
[0.303, -0.506, 1.003, 1.527, 1.234, 0.639, 0.987] ,
[0.307, -0.503, 1.002, 1.527, 1.235, 0.638, 0.981] ,
[0.31, -0.501, 1.002, 1.527, 1.236, 0.636, 0.975] ,
[0.312, -0.499, 1.002, 1.527, 1.237, 0.635, 0.971] ,
[0.314, -0.498, 1.001, 1.527, 1.237, 0.635, 0.968] ,
[0.315, -0.497, 1.001, 1.527, 1.237, 0.634, 0.966] ,
[0.315, -0.497, 1.001, 1.527, 1.238, 0.634, 0.966] ]



    point_sum = len(joint_angles_goal_list)
    point_now = 0
    mse = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    # joint_angles_now = pose_init
    joint_angles_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # joint_velocities_now = udp.controller.limb.joint_velocities()
    joint_velocities_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    z1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    dy_tau = udp.controller.limb.joint_efforts()

    cur_time = rospy.get_time()
    pre_time = cur_time

    Time = 10
    count = 1500
    ratio = count / Time
    step_size = Time / count
    output_size = 100
    out_ratio = count / output_size


    '''作图用'''
    '''关节空间'''
    joint_effort_display = np.zeros((7, output_size+1), dtype=float)
    joint_actual_pos_display = np.zeros((7, output_size+1), dtype=float)
    joint_req_pos_display = np.zeros((7, output_size+1), dtype=float)
    tout = np.zeros((7, output_size+1), dtype=float)
    xyz_display = np.zeros((3, output_size + 1), dtype=float)
    xyz_req_display = np.zeros((3, output_size + 1), dtype=float)
    sat_display = np.zeros((1, output_size + 1), dtype=float)


    '''输出用计数'''
    a = 0
    cur_time = rospy.get_time()
    pre_time = cur_time

    # temp = 0
    # for key in joint_angles_goal[0]:
    #     joint_angles_goal_list[temp] = joint_angles_goal[0][key]
    #     temp += 1


    for i in range(0, count):
        if not rospy.is_shutdown():
            '''得到角度'''
            joint_angles_now = udp.controller.limb.joint_angles()
            temp = 0
            for key in joint_angles_now:
                joint_angles_now_list[temp] = joint_angles_now[key]
                temp = temp + 1

            '''得到速度'''
            joint_velocities_now = udp.controller.limb.joint_velocities()
            temp = 0
            for key in joint_velocities_now:
                joint_velocities_now_list[temp] = joint_velocities_now[key]
                temp = temp + 1


            '''计算出当前应该的目标点'''
            if point_now < point_sum:
                if i%8 == 0:
                    point_now = point_now + 1
            # if point_now < point_sum:
            #     if abs(joint_angles_goal_list[point_now][0] - joint_angles_now_list[0]) < 0.1 and \
            #         abs(joint_angles_goal_list[point_now][1] - joint_angles_now_list[1]) < 0.1 and\
            #         abs(joint_angles_goal_list[point_now][2] - joint_angles_now_list[2]) < 0.1 and\
            #         abs(joint_angles_goal_list[point_now][6] - joint_angles_now_list[6]) < 0.1:
            #         point_now = point_now + 1

            dy_tau = udp.controller.torquecommand(joint_angles_goal_list[point_now-1])
            '''计算z1'''
            for aaa in range(0, 7):
                z1[aaa] = joint_angles_now_list[aaa] - joint_angles_goal_list[point_now-1][aaa]
                '''计算MSE'''
                mse[aaa] = mse[aaa] + 0.5 * abs(z1[aaa]) * abs(z1[aaa])
            print mse
           #
           #  '''计算alpha'''
           #  for aaa in range(0, 7):
           #      alpha[aaa] = -udp.controller.torController[aaa].get_kp() * z1[aaa] + joint_vel_goal_list[point_now-1][aaa]
           #
           #  '''计算z2'''
           #  for aaa in range(0, 7):
           #      z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]
           #      udp.trans_z2_list[aaa] = z2[aaa]
           #
           # # print z2
           #  '''得到通信计算的值'''
           #  temp = 0
           #  for key in dy_tau:
           #      dy_tau[key] = -z1[temp] - udp.controller.torController[temp].get_kd() * z2[temp] + udp.real_thread_result[temp]
           #      temp = temp + 1
            #print dy_tau

                # if key == "right_s0" or key == "right_s1" or key == "right_e0" or key == "right_e1":
                #     if dy_tau[key] > 20:
                #         dy_tau[key] = 20
                #     elif dy_tau[key] < -20:
                #         dy_tau[key] = -20
                #     else:
                #         pass
                # else:
                #     if dy_tau[key] > 12:
                #         dy_tau[key] = 12
                #     elif dy_tau[key] < -12:
                #         dy_tau[key] = -12
                #     else:
                #         pass
            sat = 2.0
            if a == 0:
                temp = 0
                start_time = rospy.get_time()
                get_pose = udp.controller.limb.joint_angles()

                xyz_pose = udp.controller.limb.endpoint_pose()
                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.4
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4
                sat_display[a, 0] = sat
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]
                    #joint_effort_display[temp,a] = tau[(0,temp)]
                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now-1][temp]
                    tout[temp, a] = 0
                    temp = temp + 1
                a = a + 1
            print point_now
            # print dy_tau

            udp.controller.limb.set_joint_torques(dy_tau)
            #udp.controller.limb.move_to_joint_positions(joint_angles_goal)

            '''作图用'''
            if i % out_ratio == 0:
                display_cur_time = rospy.get_time()
                '''关节角度 '''
                temp = 0
                get_pose = udp.controller.limb.joint_angles()
                xyz_pose = udp.controller.limb.endpoint_pose()

                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.4 + 0.004*a
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
                    # joint_effort_display[temp,a] = tau[(0,temp)]
                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now-1][temp]
                    tout[temp, a] = float(display_cur_time - start_time)
                    temp = temp + 1
                a = a + 1
            Rate.sleep()

        rospy.on_shutdown(udp.controller.shutdown_close)
    udp.thread_stop = True
    udp.controller.limb.exit_control_mode()
    # udp.controller.limb.move_to_neutral()





    #tout = np.linspace(0, 10, output_size+1)
    '''关节空间'''

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
    plt.legend(loc = 7, fontsize=legend_size)
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
    plt.legend(loc=1, fontsize=legend_size)
    plt.savefig('picture/right/right_endpoint_position.eps', format='eps')


    # plt.show()
if __name__ == '__main__':
    main()