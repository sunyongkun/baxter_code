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
        t2 = [-1.05, 1.245, -0.677, -1.134, 2.358, 0.472, -0.935]
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

        # w0
        self.torController[0].set_kp(50)  # 130#80.0#*0.6
        self.torController[0].set_ki(0.2)
        self.torController[0].set_kd(2.5)  # 10#15#0.01#*0.6#21.0

        # w1
        self.torController[1].set_kp(60)  # 130#80.0#*0.6
        self.torController[1].set_ki(0.2)
        self.torController[1].set_kd(1.3)  # 10#15#0.01#*0.6#21.0

        #w2
        self.torController[2].set_kp(5.1)
        self.torController[2].set_ki(0.1)  # 0.1
        self.torController[2].set_kd(2.5)

        # e0
        self.torController[3].set_kp(14)  # 130#80.0#*0.6
        self.torController[3].set_ki(0.2)  # 0.05
        self.torController[3].set_kd(3)  # 10#15#0.01#*0.6#21.0

        # e1
        self.torController[4].set_kp(25)  # 130#80.0#*0.6
        self.torController[4].set_ki(0.2)
        self.torController[4].set_kd(3)  # 10#15#0.01#*0.6#21.0

        #s0
        self.torController[5].set_kp(35)  # 12
        self.torController[5].set_ki(4)
        self.torController[5].set_kd(4)  # 10

        #s1
        self.torController[6].set_kp(15)  # 130#80.0#*0.6
        self.torController[6].set_ki(2)
        self.torController[6].set_kd(4)  # 10

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

    udp = from_udp('left')
    endpoint_pose_init = udp.controller.limb.endpoint_pose()
    endpoint_pose = endpoint_pose_init['position']

    pose_init = udp.controller.limb.joint_angles()
    print pose_init

    joint_goal_init = udp.controller.limb.joint_angles()
    joint_angles_goal_list = [[-1.05, 1.245, -0.677, -1.134, 2.358, 0.472, -0.935] ,
[-1.05, 1.246, -0.678, -1.133, 2.357, 0.471, -0.934] ,
[-1.05, 1.246, -0.68, -1.133, 2.354, 0.47, -0.933] ,
[-1.049, 1.248, -0.683, -1.131, 2.349, 0.467, -0.931] ,
[-1.048, 1.25, -0.687, -1.13, 2.343, 0.464, -0.928] ,
[-1.046, 1.252, -0.692, -1.128, 2.335, 0.46, -0.924] ,
[-1.045, 1.254, -0.698, -1.125, 2.325, 0.456, -0.919] ,
[-1.043, 1.258, -0.704, -1.123, 2.314, 0.451, -0.914] ,
[-1.041, 1.261, -0.712, -1.12, 2.301, 0.445, -0.908] ,
[-1.039, 1.264, -0.72, -1.116, 2.288, 0.438, -0.902] ,
[-1.036, 1.268, -0.728, -1.113, 2.273, 0.431, -0.896] ,
[-1.034, 1.272, -0.737, -1.109, 2.258, 0.424, -0.889] ,
[-1.031, 1.276, -0.745, -1.105, 2.242, 0.416, -0.882] ,
[-1.029, 1.281, -0.755, -1.101, 2.225, 0.408, -0.875] ,
[-1.026, 1.285, -0.764, -1.097, 2.208, 0.4, -0.867] ,
[-1.024, 1.289, -0.773, -1.093, 2.19, 0.391, -0.86] ,
[-1.021, 1.294, -0.782, -1.089, 2.172, 0.382, -0.853] ,
[-1.019, 1.298, -0.791, -1.085, 2.154, 0.374, -0.846] ,
[-1.016, 1.302, -0.799, -1.081, 2.136, 0.365, -0.839] ,
[-1.014, 1.306, -0.807, -1.076, 2.118, 0.356, -0.832] ,
[-1.012, 1.31, -0.815, -1.072, 2.1, 0.347, -0.826] ,
[-1.009, 1.314, -0.823, -1.068, 2.082, 0.338, -0.819] ,
[-1.007, 1.318, -0.831, -1.064, 2.064, 0.329, -0.813] ,
[-1.005, 1.322, -0.838, -1.06, 2.046, 0.32, -0.807] ,
[-1.003, 1.326, -0.845, -1.056, 2.028, 0.31, -0.801] ,
[-1.001, 1.33, -0.853, -1.052, 2.01, 0.301, -0.795] ,
[-0.999, 1.334, -0.86, -1.048, 1.992, 0.292, -0.789] ,
[-0.997, 1.337, -0.867, -1.044, 1.975, 0.283, -0.783] ,
[-0.995, 1.341, -0.873, -1.04, 1.957, 0.274, -0.777] ,
[-0.993, 1.345, -0.88, -1.036, 1.939, 0.265, -0.771] ,
[-0.991, 1.348, -0.887, -1.032, 1.921, 0.256, -0.765] ,
[-0.99, 1.352, -0.894, -1.028, 1.903, 0.246, -0.759] ,
[-0.988, 1.355, -0.901, -1.024, 1.886, 0.237, -0.753] ,
[-0.986, 1.359, -0.908, -1.02, 1.868, 0.228, -0.747] ,
[-0.984, 1.363, -0.915, -1.016, 1.851, 0.22, -0.741] ,
[-0.982, 1.366, -0.922, -1.012, 1.833, 0.211, -0.735] ,
[-0.98, 1.37, -0.929, -1.008, 1.816, 0.202, -0.729] ,
[-0.978, 1.374, -0.936, -1.004, 1.798, 0.193, -0.723] ,
[-0.976, 1.377, -0.943, -1.0, 1.781, 0.184, -0.716] ,
[-0.974, 1.381, -0.951, -0.996, 1.764, 0.176, -0.71] ,
[-0.972, 1.385, -0.958, -0.991, 1.747, 0.167, -0.704] ,
[-0.97, 1.388, -0.965, -0.987, 1.73, 0.158, -0.697] ,
[-0.968, 1.392, -0.973, -0.983, 1.713, 0.15, -0.691] ,
[-0.966, 1.395, -0.98, -0.979, 1.696, 0.141, -0.685] ,
[-0.964, 1.399, -0.987, -0.975, 1.679, 0.133, -0.678] ,
[-0.962, 1.403, -0.995, -0.971, 1.662, 0.124, -0.672] ,
[-0.96, 1.406, -1.002, -0.967, 1.645, 0.116, -0.666] ,
[-0.958, 1.41, -1.01, -0.963, 1.628, 0.107, -0.659] ,
[-0.955, 1.414, -1.017, -0.959, 1.611, 0.099, -0.653] ,
[-0.953, 1.417, -1.024, -0.955, 1.595, 0.09, -0.647] ,
[-0.951, 1.421, -1.032, -0.951, 1.578, 0.082, -0.64] ,
[-0.949, 1.424, -1.039, -0.948, 1.561, 0.073, -0.634] ,
[-0.947, 1.428, -1.046, -0.944, 1.545, 0.065, -0.628] ,
[-0.944, 1.432, -1.054, -0.94, 1.528, 0.056, -0.622] ,
[-0.942, 1.435, -1.061, -0.936, 1.512, 0.048, -0.616] ,
[-0.94, 1.439, -1.068, -0.932, 1.495, 0.039, -0.61] ,
[-0.937, 1.442, -1.075, -0.929, 1.479, 0.031, -0.604] ,
[-0.935, 1.446, -1.082, -0.925, 1.462, 0.022, -0.598] ,
[-0.933, 1.45, -1.09, -0.921, 1.446, 0.014, -0.592] ,
[-0.93, 1.453, -1.097, -0.917, 1.43, 0.005, -0.586] ,
[-0.928, 1.457, -1.104, -0.914, 1.413, -0.003, -0.58] ,
[-0.925, 1.46, -1.111, -0.91, 1.397, -0.011, -0.574] ,
[-0.923, 1.464, -1.118, -0.906, 1.381, -0.019, -0.568] ,
[-0.921, 1.467, -1.126, -0.903, 1.365, -0.028, -0.563] ,
[-0.918, 1.471, -1.133, -0.899, 1.349, -0.036, -0.557] ,
[-0.916, 1.475, -1.14, -0.895, 1.333, -0.044, -0.551] ,
[-0.914, 1.478, -1.147, -0.891, 1.317, -0.052, -0.545] ,
[-0.912, 1.482, -1.154, -0.888, 1.301, -0.059, -0.539] ,
[-0.909, 1.486, -1.162, -0.884, 1.286, -0.067, -0.533] ,
[-0.907, 1.489, -1.169, -0.88, 1.27, -0.075, -0.527] ,
[-0.905, 1.493, -1.176, -0.876, 1.254, -0.082, -0.521] ,
[-0.903, 1.496, -1.184, -0.873, 1.238, -0.09, -0.515] ,
[-0.901, 1.5, -1.191, -0.869, 1.223, -0.098, -0.509] ,
[-0.898, 1.504, -1.198, -0.865, 1.207, -0.105, -0.503] ,
[-0.896, 1.507, -1.205, -0.861, 1.191, -0.113, -0.497] ,
[-0.894, 1.511, -1.213, -0.857, 1.175, -0.121, -0.491] ,
[-0.892, 1.514, -1.22, -0.853, 1.159, -0.128, -0.485] ,
[-0.89, 1.518, -1.227, -0.85, 1.143, -0.136, -0.479] ,
[-0.888, 1.521, -1.234, -0.846, 1.126, -0.144, -0.473] ,
[-0.886, 1.525, -1.241, -0.842, 1.11, -0.152, -0.466] ,
[-0.884, 1.528, -1.248, -0.838, 1.093, -0.161, -0.46] ,
[-0.882, 1.531, -1.255, -0.834, 1.076, -0.169, -0.454] ,
[-0.88, 1.535, -1.262, -0.83, 1.059, -0.178, -0.447] ,
[-0.878, 1.538, -1.269, -0.826, 1.041, -0.187, -0.441] ,
[-0.877, 1.541, -1.276, -0.821, 1.023, -0.196, -0.434] ,
[-0.875, 1.544, -1.282, -0.817, 1.005, -0.205, -0.427] ,
[-0.873, 1.547, -1.289, -0.813, 0.987, -0.215, -0.42] ,
[-0.871, 1.55, -1.295, -0.809, 0.969, -0.224, -0.414] ,
[-0.869, 1.553, -1.302, -0.805, 0.951, -0.233, -0.407] ,
[-0.868, 1.555, -1.307, -0.801, 0.934, -0.243, -0.401] ,
[-0.866, 1.558, -1.313, -0.798, 0.917, -0.251, -0.395] ,
[-0.865, 1.56, -1.318, -0.794, 0.902, -0.26, -0.389] ,
[-0.863, 1.562, -1.323, -0.791, 0.887, -0.268, -0.384] ,
[-0.862, 1.564, -1.327, -0.788, 0.873, -0.275, -0.379] ,
[-0.861, 1.566, -1.331, -0.785, 0.86, -0.282, -0.374] ,
[-0.86, 1.568, -1.335, -0.783, 0.849, -0.288, -0.37] ,
[-0.859, 1.569, -1.338, -0.781, 0.84, -0.293, -0.367] ,
[-0.859, 1.57, -1.34, -0.779, 0.833, -0.297, -0.364] ,
[-0.858, 1.571, -1.342, -0.778, 0.827, -0.3, -0.362] ,
[-0.858, 1.571, -1.343, -0.777, 0.823, -0.302, -0.361] ,
[-0.858, 1.571, -1.343, -0.777, 0.822, -0.303, -0.36] ]
    joint_vel_goal_list =[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[0.016, 0.023, 0.172, -0.036, 0.06, 0.206, -0.0] ,
[0.033, 0.036, 0.293, -0.06, 0.1, 0.363, -0.0] ,
[0.05, 0.038, 0.365, -0.074, 0.12, 0.471, -0.01] ,
[0.069, 0.029, 0.388, -0.078, 0.121, 0.53, -0.0] ,
[0.088, 0.009, 0.36, -0.07, 0.1, 0.54, -0.0] ,
[0.101, -0.012, 0.322, -0.059, 0.077, 0.53, 0.0] ,
[0.103, -0.023, 0.311, -0.053, 0.067, 0.53, 0.004] ,
[0.094, -0.025, 0.328, -0.053, 0.07, 0.541, 0.0] ,
[0.073, -0.018, 0.374, -0.057, 0.086, 0.561, 0.0] ,
[0.041, -0.001, 0.447, -0.066, 0.116, 0.591, 0.0] ,
[0.009, 0.016, 0.522, -0.076, 0.148, 0.624, 0.0] ,
[-0.009, 0.026, 0.575, -0.086, 0.17, 0.654, 0.0] ,
[-0.015, 0.027, 0.603, -0.093, 0.184, 0.679, 0.003] ,
[-0.007, 0.021, 0.608, -0.1, 0.188, 0.701, 0.0] ,
[0.013, 0.007, 0.59, -0.105, 0.184, 0.719, -0.0] ,
[0.036, -0.008, 0.564, -0.108, 0.176, 0.734, -0.0] ,
[0.052, -0.016, 0.544, -0.11, 0.171, 0.746, -0.005] ,
[0.061, -0.018, 0.532, -0.109, 0.169, 0.755, -0.005] ,
[0.062, -0.014, 0.527, -0.106, 0.169, 0.762, -0.004] ,
[0.056, -0.003, 0.529, -0.101, 0.173, 0.765, -0.002] ,
[0.048, 0.009, 0.535, -0.097, 0.176, 0.768, 0.0] ,
[0.043, 0.015, 0.541, -0.094, 0.178, 0.771, 0.001] ,
[0.042, 0.017, 0.548, -0.094, 0.178, 0.776, 0.001] ,
[0.045, 0.014, 0.554, -0.095, 0.176, 0.782, 0.0] ,
[0.051, 0.005, 0.562, -0.099, 0.173, 0.789, -0.002] ,
[0.057, -0.004, 0.567, -0.104, 0.169, 0.796, -0.004] ,
[0.059, -0.009, 0.568, -0.107, 0.167, 0.801, -0.0] ,
[0.056, -0.012, 0.566, -0.109, 0.167, 0.804, -0.0] ,
[0.05, -0.011, 0.559, -0.11, 0.168, 0.805, -0.004] ,
[0.039, -0.007, 0.548, -0.11, 0.17, 0.805, -0.002] ,
[0.028, -0.002, 0.539, -0.11, 0.173, 0.804, 0.0] ,
[0.023, 0.003, 0.535, -0.11, 0.174, 0.804, 0.001] ,
[0.022, 0.006, 0.537, -0.111, 0.174, 0.806, 0.001] ,
[0.026, 0.009, 0.545, -0.113, 0.172, 0.808, 0.0] ,
[0.035, 0.012, 0.559, -0.114, 0.169, 0.813, -0.003] ,
[0.045, 0.013, 0.571, -0.116, 0.165, 0.815, -0.006] ,
[0.051, 0.011, 0.576, -0.115, 0.162, 0.813, -0.0] ,
[0.054, 0.008, 0.573, -0.112, 0.159, 0.806, -0.0] ,
[0.055, 0.002, 0.562, -0.106, 0.157, 0.794, -0.0] ,
[0.052, -0.005, 0.543, -0.099, 0.155, 0.777, -0.01] ,
[0.047, -0.01, 0.52, -0.092, 0.152, 0.756, -0.0] ,
[0.043, -0.006, 0.499, -0.087, 0.149, 0.732, -0.009] ,
[0.04, 0.006, 0.48, -0.085, 0.145, 0.703, -0.007] ,
[0.037, 0.027, 0.462, -0.085, 0.139, 0.67, -0.006] ,
[0.035, 0.056, 0.446, -0.087, 0.132, 0.633, -0.004] ,
[0.032, 0.081, 0.427, -0.089, 0.126, 0.596, -0.002] ,
[0.029, 0.091, 0.398, -0.089, 0.12, 0.561, 0.0] ,
[0.026, 0.084, 0.361, -0.085, 0.115, 0.529, 0.001] ,
[0.021, 0.062, 0.315, -0.079, 0.112, 0.5, 0.002] ,
[0.016, 0.023, 0.261, -0.069, 0.109, 0.473, 0.003] ,
[0.011, -0.02, 0.208, -0.059, 0.106, 0.445, 0.004] ,
[0.008, -0.056, 0.169, -0.049, 0.099, 0.414, 0.004] ,
[0.005, -0.086, 0.144, -0.041, 0.09, 0.378, 0.004] ,
[0.004, -0.109, 0.132, -0.034, 0.078, 0.337, 0.004] ,
[0.005, -0.126, 0.134, -0.027, 0.063, 0.293, 0.004] ,
[0.006, -0.133, 0.141, -0.022, 0.048, 0.249, 0.004] ,
[0.007, -0.127, 0.145, -0.02, 0.038, 0.213, 0.004] ,
[0.008, -0.11, 0.145, -0.021, 0.031, 0.182, 0.004] ,
[0.009, -0.08, 0.142, -0.024, 0.029, 0.159, 0.004] ,
[0.011, -0.039, 0.136, -0.03, 0.031, 0.141, 0.005] ,
[0.012, 0.001, 0.126, -0.036, 0.034, 0.128, 0.005] ,
[0.012, 0.026, 0.112, -0.04, 0.036, 0.116, 0.005] ,
[0.012, 0.034, 0.094, -0.041, 0.036, 0.105, 0.004] ,
[0.011, 0.026, 0.073, -0.04, 0.035, 0.095, 0.002] ,
[0.01, 0.003, 0.048, -0.037, 0.032, 0.086, 0.0] ,
[0.009, -0.023, 0.025, -0.033, 0.029, 0.077, -0.003] ,
[0.008, -0.036, 0.009, -0.031, 0.027, 0.069, -0.004] ,
[0.007, -0.039, 0.001, -0.029, 0.025, 0.061, -0.005] ,
[0.006, -0.029, -0.001, -0.028, 0.025, 0.053, -0.005] ,
[0.005, -0.008, 0.005, -0.029, 0.024, 0.045, -0.004] ,
[0.005, 0.014, 0.013, -0.029, 0.024, 0.037, -0.002] ,
[0.004, 0.027, 0.017, -0.027, 0.022, 0.03, -0.001] ,
[0.003, 0.029, 0.019, -0.023, 0.019, 0.023, 0.001] ,
[0.003, 0.023, 0.018, -0.017, 0.014, 0.016, 0.002] ,
[0.002, 0.006, 0.013, -0.009, 0.009, 0.01, 0.004] ,
[0.002, -0.012, 0.008, -0.002, 0.004, 0.005, 0.005] ,
[0.001, -0.021, 0.004, 0.002, 0.003, 0.001, 0.005] ,
[0.0, -0.023, 0.001, 0.003, 0.006, -0.003, 0.004] ,
[-0.002, -0.018, 0.0, 0.001, 0.012, -0.005, 0.002] ,
[-0.003, -0.004, 0.0, -0.003, 0.021, -0.005, -0.001] ,
[-0.005, 0.01, 0.0, -0.008, 0.029, -0.005, -0.004] ,
[-0.005, 0.017, 0.0, -0.011, 0.031, -0.005, -0.005] ,
[-0.004, 0.018, 0.0, -0.01, 0.026, -0.004, -0.005] ,
[-0.003, 0.012, 0.0, -0.006, 0.016, -0.002, -0.003] ,
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
[0.0, -0.0, 0.0, 0.009, -0.0, 0.003, 0.005] ,
[0.0, -0.0, -0.001, 0.0, -0.0, 0.006, 0.0] ,
[0.0, -0.0, -0.001, 0.0, -0.0, 0.01, 0.02] ,
[0.0, -0.0, -0.002, 0.0, -0.0, 0.0, 0.03] ,
[0.0, -0.0, -0.002, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.007, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.01, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
[0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0] ,
]


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
    plt.plot(tout[0].T, -1 * sat_display[0], 'm--', linewidth=linesize)
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
    plt.savefig('picture/left/left_endpoint_position.eps', format='eps')

    # plt.show()
if __name__ == '__main__':
    main()