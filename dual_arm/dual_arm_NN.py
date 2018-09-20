#!/usr/bin/env python
# coding=utf-8

import math
import sys
import numpy as np
import baxter_interface
from baxter_interface import CHECK_VERSION
import rospy

import matplotlib.pyplot as plt
import threading
from mpl_toolkits.mplot3d import Axes3D
import moveit_commander
import geometry_msgs.msg

x_move = 0.2

class moveit_trajectory(object):
    def __init__(self, Armname):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(Armname +"_arm")
        self.limb = baxter_interface.Limb(Armname)

    def plan_target(self, des_position):
        self.group.clear_pose_targets()

        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.x = des_position['orientation'][0]
        pose_target.orientation.y = des_position['orientation'][1]
        pose_target.orientation.z = des_position['orientation'][2]
        pose_target.orientation.w = des_position['orientation'][3]
        pose_target.position.x = des_position['position'][0]
        pose_target.position.y = des_position['position'][1]
        pose_target.position.z = des_position['position'][2]
        self.group.set_pose_target(pose_target)
        plan = self.group.plan()
        # print plan

        self.get_trajectory_times = list()
        self.get_trajectory_positions = list()
        self.get_trajectory_velocities = list()
        print len(plan.joint_trajectory.points)
        for i in range(0, len(plan.joint_trajectory.points)):
            self.get_trajectory_positions.append(plan.joint_trajectory.points[i].positions)
            self.get_trajectory_velocities.append(plan.joint_trajectory.points[i].velocities)
            self.get_trajectory_times.append(0.00 + plan.joint_trajectory.points[i].time_from_start.secs + plan.joint_trajectory.points[i].time_from_start.nsecs/1000000000.0)
            print 0.00 + plan.joint_trajectory.points[i].time_from_start.secs + plan.joint_trajectory.points[i].time_from_start.nsecs/1000000000.0
            print plan.joint_trajectory.points[i].positions
            print self.get_trajectory_positions
        # print "aaa"

class cubicSpline(object):
    def __init__(self):
        self.outputsize = 101
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.f = []
        self.bt = []
        self.gm = []
        self.h = []
        self.x_sample = []
        self.y_sample = []
        self.M = []
        self.sample_count = 0
        self.bound1 = 0
        self.bound2 = 0
        self.result = []

    def initParam(self, count):
        self.a = np.zeros(count, dtype="double")
        self.b = np.zeros(count, dtype="double")
        self.c = np.zeros(count, dtype="double")
        self.d = np.zeros(count, dtype="double")
        self.f = np.zeros(count, dtype="double")
        self.bt = np.zeros(count, dtype="double")
        self.gm = np.zeros(count, dtype="double")
        self.h = np.zeros(count, dtype="double")
        self.M = np.zeros(count, dtype="double")


    def loadData(self, x_data, y_data, count, bound1, bound2):
        if len(x_data) == 0 or len(y_data) == 0 or count < 3:
            return False

        self.initParam(count)

        self.x_sample = x_data
        self.y_sample = y_data
        self.sample_count = count
        self.bound1 = bound1
        self.bound2 = bound2

    def spline(self):

        f1 = self.bound1
        f2 = self.bound2

        for i in range(0, self.sample_count):
            self.b[i] = 2
        for i in range(0, self.sample_count - 1):
            self.h[i] = self.x_sample[i+1] - self.x_sample[i]
        for i in range(0, self.sample_count - 1):
            self.a[i] = self.h[i-1]/(self.h[i-1] + self.h[i])
        self.a[self.sample_count - 1] = 1

        self.c[0] = 1
        for i in range(1, self.sample_count-1):
            self.c[i] = self.h[i] / (self.h[i-1] + self.h[i])



        for i in range(0, self.sample_count-1):
            self.f[i] = (self.y_sample[i+1]-self.y_sample[i])/(self.x_sample[i+1]-self.x_sample[i])

        for i in range(1, self.sample_count-1):
            self.d[i] = 6*(self.f[i]-self.f[i-1])/(self.h[i-1]+self.h[i])

        """追赶法解方程"""
        self.d[0] = 6*(self.f[0] - f1)/self.h[0]
        self.d[self.sample_count - 1] = 6*(f2 - self.f[self.sample_count-2])/self.h[self.sample_count-2]

        self.bt[0] = self.c[0]/self.b[0]
        for i in range(1, self.sample_count-1):
            self.bt[i] = self.c[i]/(self.b[i] - self.a[i]*self.bt[i-1])

        self.gm[0] = self.d[0]/self.b[0]
        for i in range(1, self.sample_count):
            self.gm[i] = (self.d[i] - self.a[i]*self.gm[i-1])/(self.b[i]-self.a[i]*self.bt[i-1])
        self.M[self.sample_count-1] = self.gm[self.sample_count-1]
        temp = self.sample_count - 2
        for i in range(0, self.sample_count-1):
            self.M[temp] = self.gm[temp] - self.bt[temp]*self.M[temp+1]
            temp = temp - 1


    def getYbyX(self, x_in):
        klo = 0
        khi = self.sample_count - 1
        """二分法查找x所在的区间段"""
        while (khi - klo) >1:
            k = (khi+klo)/2
            if self.x_sample[k] > x_in:
                khi = k
            else:
                klo = k
        hh = self.x_sample[khi] - self.x_sample[klo]
        aa = (self.x_sample[khi] - x_in)/hh
        bb = (x_in - self.x_sample[klo])/hh

        y_out = aa * self.y_sample[klo] + bb * self.y_sample[khi] + \
                ((aa*aa*aa-aa)*self.M[klo] + (bb*bb*bb-bb)*self.M[khi])*hh*hh/6.0
        vel = self.M[khi] * (x_in - self.x_sample[klo]) * (x_in - self.x_sample[klo]) / (2 * hh)\
            - self.M[klo] * (self.x_sample[khi] - x_in) * (self.x_sample[khi] - x_in) / (2 * hh)\
            + (self.y_sample[khi] - self.y_sample[klo]) / hh - hh * (self.M[khi] - self.M[klo]) / 6

        return y_out, vel

    '''all_y_data 为得到角度后的转置，所以all_y_data[0]为第一个关节的所有的角度'''
    def caculate(self, all_x_data, all_y_data):
        length = len(all_x_data)

        dis = (all_x_data[length - 1] - all_x_data[0]) / (self.outputsize - 1)
        self.pos_result = np.zeros((self.outputsize, 7), dtype="double")
        self.vel_result = np.zeros((self.outputsize, 7), dtype="double")
        for ii in range(0, 7):
            self.loadData(all_x_data, all_y_data[ii], length, 0, 0)
            self.spline()
            x_out = -dis
            for i in range(0, self.outputsize):
                x_out = x_out + dis
                self.pos_result[i][ii], self.vel_result[i][ii] = self.getYbyX(x_out)

def left_arm_control():
    """ 获取期望关节角和速度 """
    Rate = rospy.Rate(200)
    left_arm = baxter_interface.Limb("left")
    moveit = moveit_trajectory("left")
    insert = cubicSpline()

    pose_init = left_arm.endpoint_pose()

    des_position = dict(position=[1, 2, 3], orientation = [1, 2, 3, 4])
    des_position['position'][0] = pose_init['position'][0] + x_move
    des_position['position'][1] = pose_init['position'][1]
    des_position['position'][2] = pose_init['position'][2]
    des_position['orientation'][0] = pose_init['orientation'][0]
    des_position['orientation'][1] = pose_init['orientation'][1]
    des_position['orientation'][2] = pose_init['orientation'][2]
    des_position['orientation'][3] = pose_init['orientation'][3]
    moveit.plan_target(des_position)

    joint_angles_goal_list = list()
    joint_vel_goal_list = list()

    for foo in range(0, len(moveit.get_trajectory_positions)):
        joint_angles_goal_list.append(list(moveit.get_trajectory_positions[foo]))

    for foo in range(0, len(joint_angles_goal_list)):
        joint_angles_goal_list[foo][0], joint_angles_goal_list[foo][4] = joint_angles_goal_list[foo][4], \
                                                                         joint_angles_goal_list[foo][0]
        joint_angles_goal_list[foo][1], joint_angles_goal_list[foo][5] = joint_angles_goal_list[foo][5], \
                                                                         joint_angles_goal_list[foo][1]
        joint_angles_goal_list[foo][2], joint_angles_goal_list[foo][6] = joint_angles_goal_list[foo][6], \
                                                                         joint_angles_goal_list[foo][2]
        joint_angles_goal_list[foo][3], joint_angles_goal_list[foo][6] = joint_angles_goal_list[foo][6], \
                                                                         joint_angles_goal_list[foo][3]
        joint_angles_goal_list[foo][4], joint_angles_goal_list[foo][6] = joint_angles_goal_list[foo][6], \
                                                                         joint_angles_goal_list[foo][4]
        joint_angles_goal_list[foo][5], joint_angles_goal_list[foo][6] = joint_angles_goal_list[foo][6], \
                                                                         joint_angles_goal_list[foo][5]
        cacul = np.mat(joint_angles_goal_list).T
        data_format_trans = np.array(cacul)
        insert.caculate(moveit.get_trajectory_times, data_format_trans)

    # joint_angles_goal_list = insert.pos_result
    # joint_vel_goal_list = insert.vel_result

    """ 初始化变量 """
    point_sum = len(insert.pos_result)
    point_now = 0
    joint_angles_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_velocities_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 速度误差
    z2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dy_tau = left_arm.joint_efforts()  # 实际关节转矩

    var = 8
    count = var * (point_sum)  # 迭代次数
    out_ratio = 4 # 作图抽取率
    output_size = count / out_ratio

    """ 画图用初始化，输入关节力，关节实际位置，关节期望位置，时间，末端实际位置，末端期望位置 """
    joint_effort_display = np.zeros((7, output_size), dtype=float)
    joint_actual_pos_display = np.zeros((7, output_size), dtype=float)
    joint_req_pos_display = np.zeros((7, output_size), dtype=float)
    tout = np.zeros((output_size), dtype=float)
    xyz_display = np.zeros((3, output_size), dtype=float)
    xyz_req_display = np.zeros((3, output_size), dtype=float)

    # K1,K2 w0 w1 w2 e0 e1 s0 s1
    K1 = np.array([15.7, 22, 20.3, 12.6, 15.0, 17.7, 20.0])
    K2 = np.array([1.2, 2.0, 5.1, 10.1, 4.5, 2.1, 2.2])

    a = 0  # 输出用计数
    i = 0  # 循环计数

    while point_now < point_sum: # point_sum等于100，每8个迭代点算一次目标值
        if not rospy.is_shutdown():  # 循环过程中ctrl+c判断
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
            if i % var == 0:
                point_now = point_now + 1

            """ 计算z1,计算关节角误差，上面8个+1，因此这里计算当前-1 """
            for aaa in range(0, 7):
                z1[aaa] = joint_angles_now_list[aaa] - insert.pos_result[point_now - 1][aaa]

            """ 计算alpha """
            for aaa in range(0, 7):
                alpha[aaa] = -K1[aaa] * z1[aaa] + insert.vel_result[point_now - 1][aaa]

            """ 计算z2 """
            for aaa in range(0, 7):
                z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]

            """ 得到通信计算的值，即神经网络部分的值,udp_l.controller.torController[temp].get_kd()只是7个数 """
            temp = 0
            for key in dy_tau:
                dy_tau[key] = -z1[temp] - K2[aaa] * z2[temp]
                # if dy_tau[key] >
                temp = temp + 1

            """ 画图初始化 """
            if a == 0:
                temp = 0
                tout[a] = 0
                start_time = rospy.get_time()
                xyz_pose = left_arm.endpoint_pose()
                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.5
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4

                get_pose = left_arm.joint_angles()
                # 把字典转换成序列
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]

                    joint_req_pos_display[temp, a] = insert.pos_result[point_now - 1][temp]
                    temp = temp + 1
                a = a + 1

            left_arm.set_joint_torques(dy_tau)

            """ 作图用 """
            if i % out_ratio == 0:
                display_cur_time = rospy.get_time()  # 控制后的时间
                """ 关节角度 """
                temp = 0
                get_pose = left_arm.joint_angles()
                # print get_pose
                xyz_pose = left_arm.endpoint_pose()
                tout[a] = display_cur_time - start_time

                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.5 + 0.002 * a
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]

                    joint_req_pos_display[temp, a] = insert.pos_result[point_now - 1][temp]
                    temp = temp + 1

                a = a + 1
            i = i + 1
            Rate.sleep()
    left_arm.exit_control_mode()

    plt.subplot(1, 1, 1)
    plt.title("left_joint_w0")
    plt.plot(tout.T, joint_actual_pos_display[0], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[0], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[0] - joint_actual_pos_display[0], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/left/left_joint_w0.eps', format='eps')

    fig2 = plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.title("left_joint_w1")
    plt.plot(tout.T, joint_actual_pos_display[1], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[1], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[1] - joint_actual_pos_display[1], linewidth=3, color="blue",
             label="error value")
    # plt.ylim(0, 3)
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/left/left_joint_w1.eps', format='eps')

    fig3 = plt.figure(3)
    plt.subplot(1, 1, 1)
    plt.title("left_joint_w2")
    plt.plot(tout.T, joint_actual_pos_display[2], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[2], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[2] - joint_actual_pos_display[2], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/left/left_joint_w2.eps', format='eps')

    fig4 = plt.figure(4)
    plt.subplot(1, 1, 1)
    plt.title("left_joint_e0")
    plt.plot(tout.T, joint_actual_pos_display[3], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[3], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[3] - joint_actual_pos_display[3], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/left/left_joint_e0.eps', format='eps')

    fig5 = plt.figure(5)
    plt.subplot(1, 1, 1)
    plt.title("left_joint_e1")
    plt.plot(tout.T, joint_actual_pos_display[4], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[4], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[4] - joint_actual_pos_display[4], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/left/left_joint_e1.eps', format='eps')

    fig6 = plt.figure(6)
    plt.subplot(1, 1, 1)
    plt.title("left_joint_s0")
    plt.plot(tout.T, joint_actual_pos_display[5], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[5], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[5] - joint_actual_pos_display[5], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/left/left_joint_s0.eps', format='eps')

    fig7 = plt.figure(7)
    plt.subplot(1, 1, 1)
    plt.title("left_joint_s1")
    plt.plot(tout.T, joint_actual_pos_display[6], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[6], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[6] - joint_actual_pos_display[6], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/left/left_joint_s1.eps', format='eps')

    # 转矩画图
    fig8 = plt.figure(8)
    plt.subplot(1, 1, 1)
    plt.title("left_torques")
    plt.plot(tout.T, joint_effort_display[0], linewidth=3, label='joint_w0')
    plt.plot(tout.T, joint_effort_display[1], linewidth=3, label='joint_w1')
    plt.plot(tout.T, joint_effort_display[2], linewidth=3, label='joint_w2')
    plt.plot(tout.T, joint_effort_display[3], linewidth=3, label='joint_e0')
    plt.plot(tout.T, joint_effort_display[4], linewidth=3, label='joint_e1')
    plt.plot(tout.T, joint_effort_display[5], linewidth=3, label='joint_s0')
    plt.plot(tout.T, joint_effort_display[6], linewidth=3, label='joint_s1')
    plt.xlabel("time/s")
    plt.ylabel("torque/Nm")
    plt.xlim(0, 4)
    plt.legend(loc='best', bbox_to_anchor=(1,0.7))
    plt.savefig('picture/left/left_torques.eps', format='eps')

    # xyz
    fig9 = plt.figure(9)
    ax = fig9.add_subplot(111, projection='3d')
    plt.title("left_endpoint_position")
    ax.plot(xyz_display[0], xyz_display[1], xyz_display[2], label="actual value",color='red', linewidth=3)
    ax.plot(xyz_req_display[0], xyz_req_display[1], xyz_req_display[2], label="desired value", color='green', linewidth=3)
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    plt.ylim(-1, 1)
    ax.set_zlim(-1, 1.5)
    # ax.set_xlim(0, 1)
    plt.legend(loc='best', bbox_to_anchor=(1.1, 1))
    plt.savefig('picture/left/left_endpoint_position.eps', format='eps')

    plt.show()

def right_arm_control():
    """ 获取期望关节角和速度 """
    Rate = rospy.Rate(200)
    right_arm = baxter_interface.Limb("right")
    moveit = moveit_trajectory("right")
    insert = cubicSpline()

    pose_init = right_arm.endpoint_pose()

    des_position = dict(position=[1, 2, 3], orientation = [1, 2, 3, 4])
    des_position['position'][0] = pose_init['position'][0] + x_move
    des_position['position'][1] = pose_init['position'][1]
    des_position['position'][2] = pose_init['position'][2]
    des_position['orientation'][0] = pose_init['orientation'][0]
    des_position['orientation'][1] = pose_init['orientation'][1]
    des_position['orientation'][2] = pose_init['orientation'][2]
    des_position['orientation'][3] = pose_init['orientation'][3]
    moveit.plan_target(des_position)

    joint_angles_goal_list = list()
    joint_vel_goal_list = list()

    for foo in range(0, len(moveit.get_trajectory_positions)):
        joint_angles_goal_list.append(list(moveit.get_trajectory_positions[foo]))
        # joint_vel_goal_list.append(list(moveit.get_trajectory_velocities[foo]))

    for foo in range(0, len(joint_angles_goal_list)):
        joint_angles_goal_list[foo][2], joint_angles_goal_list[foo][5] = joint_angles_goal_list[foo][5], \
                                                                         joint_angles_goal_list[foo][2]
        joint_angles_goal_list[foo][3], joint_angles_goal_list[foo][6] = joint_angles_goal_list[foo][6], \
                                                                         joint_angles_goal_list[foo][3]
        joint_angles_goal_list[foo][2], joint_angles_goal_list[foo][3] = joint_angles_goal_list[foo][3], \
                                                                         joint_angles_goal_list[foo][2]
        joint_angles_goal_list[foo][2], joint_angles_goal_list[foo][4] = joint_angles_goal_list[foo][4], \
                                                                         joint_angles_goal_list[foo][2]

    cacul = np.mat(joint_angles_goal_list).T
    data_format_trans = np.array(cacul)
    insert.caculate(moveit.get_trajectory_times, data_format_trans)
    joint_angles_goal_list = insert.pos_result
    joint_vel_goal_list = insert.vel_result

    """ 初始化变量 """
    point_sum = len(insert.pos_result)
    point_now = 0
    joint_angles_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_velocities_now_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 速度误差
    z2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dy_tau = right_arm.joint_efforts()  # 实际关节转矩

    count = 8 * (point_sum)  # 迭代次数
    out_ratio = 4 # 作图抽取率
    output_size = count / out_ratio

    """ 画图用初始化，输入关节力，关节实际位置，关节期望位置，时间，末端实际位置，末端期望位置 """
    joint_effort_display = np.zeros((7, output_size), dtype=float)
    joint_actual_pos_display = np.zeros((7, output_size), dtype=float)
    joint_req_pos_display = np.zeros((7, output_size), dtype=float)
    tout = np.zeros((output_size), dtype=float)
    xyz_display = np.zeros((3, output_size), dtype=float)
    xyz_req_display = np.zeros((3, output_size), dtype=float)

    # K1,K2  s0 s1 w0 w1 w2 e0 e1
    # K1 = np.array([14.6, 22.0, 17.7, 15.0, 15.7, 10.02, 10.3])
    # K2 = np.array([2.1, 4.5, 5.1, 18.0, 1.2, 2.5, 2.1])
    K1 = np.array([14.6, 30.0, 17.7, 15.0, 15.7, 10.02, 10.3])
    K2 = np.array([2.1, 4.5, 5.1, 18.0, 1.2, 2.5, 2.1])


    a = 0  # 输出用计数
    i = 0  # 循环计数

    while point_now < point_sum: # point_sum等于100，每8个迭代点算一次目标值
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
            if i % 8 == 0:
                point_now = point_now + 1

            """ 计算z1,计算关节角误差，上面8个+1，因此这里计算当前-1 """
            for aaa in range(0, 7):
                z1[aaa] = joint_angles_now_list[aaa] - joint_angles_goal_list[point_now - 1][aaa]

            """ 计算alpha """
            for aaa in range(0, 7):
                alpha[aaa] = -K1[aaa] * z1[aaa] + joint_vel_goal_list[point_now - 1][aaa]

            """ 计算z2 """
            for aaa in range(0, 7):
                z2[aaa] = joint_velocities_now_list[aaa] - alpha[aaa]

            """ 得到通信计算的值，即神经网络部分的值,udp_l.controller.torController[temp].get_kd()只是7个数 """
            temp = 0
            for key in dy_tau:
                dy_tau[key] = -z1[temp] - K2[aaa] * z2[temp]
                temp = temp + 1

            """ 画图初始化 """
            if a == 0:
                temp = 0
                xyz_pose = right_arm.endpoint_pose()
                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.5
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4

                start_time = rospy.get_time()
                get_pose = right_arm.joint_angles()
                # print get_pose
                # 把字典转换成序列
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]

                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now - 1][temp]
                    tout[a] = 0
                    temp = temp + 1
                a = a + 1

            right_arm.set_joint_torques(dy_tau)

            """ 作图用 """
            if i % out_ratio == 0:
                display_cur_time = rospy.get_time()  # 控制后的时间
                """ 关节角度 """
                temp = 0
                get_pose = right_arm.joint_angles()
                xyz_pose = right_arm.endpoint_pose()

                xyz_display[0, a] = xyz_pose["position"][0]
                xyz_display[1, a] = xyz_pose["position"][1]
                xyz_display[2, a] = xyz_pose["position"][2]

                xyz_req_display[0, a] = 0.5 + x_move/100 * a
                xyz_req_display[1, a] = 0.0
                xyz_req_display[2, a] = 0.4
                for key in get_pose:
                    joint_actual_pos_display[temp, a] = get_pose[key]
                    joint_effort_display[temp, a] = dy_tau[key]

                    joint_req_pos_display[temp, a] = joint_angles_goal_list[point_now - 1][temp]
                    tout[a] = display_cur_time - start_time
                    temp = temp + 1

                a = a + 1
            i = i + 1
            Rate.sleep()
    right_arm.exit_control_mode()

    plt.subplot(1, 1, 1)
    plt.title("right_joint_s0")
    plt.plot(tout.T, joint_actual_pos_display[0], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[0], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[0] - joint_actual_pos_display[0], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/right/right_joint_s0.eps', format='eps')

    fig2 = plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.title("right_joint_s1")
    plt.plot(tout.T, joint_actual_pos_display[1], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[1], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[1] - joint_actual_pos_display[1], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/right/right_joint_s1.eps', format='eps')

    fig3 = plt.figure(3)
    plt.subplot(1, 1, 1)
    plt.title("right_joint_w0")
    plt.plot(tout.T, joint_actual_pos_display[2], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[2], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[2] - joint_actual_pos_display[2], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/right/right_joint_w0.eps', format='eps')

    fig4 = plt.figure(4)
    plt.subplot(1, 1, 1)
    plt.title("right_joint_w1")
    plt.plot(tout.T, joint_actual_pos_display[3], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[3], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[3] - joint_actual_pos_display[3], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/right/right_joint_w1.eps', format='eps')

    fig5 = plt.figure(5)
    plt.subplot(1, 1, 1)
    plt.title("right_joint_w2")
    plt.plot(tout.T, joint_actual_pos_display[4], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[4], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[4] - joint_actual_pos_display[4], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/right/right_joint_w2.eps', format='eps')

    fig6 = plt.figure(6)
    plt.subplot(1, 1, 1)
    plt.title("right_joint_e0")
    plt.plot(tout.T, joint_actual_pos_display[5], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[5], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[5] - joint_actual_pos_display[5], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/right/right_joint_e0.eps', format='eps')

    fig7 = plt.figure(7)
    plt.subplot(1, 1, 1)
    plt.title("right_joint_e1")
    plt.plot(tout.T, joint_actual_pos_display[6], linewidth=3, color="red", label="actual value")
    plt.plot(tout.T, joint_req_pos_display[6], linewidth=3, color="green", label="desired value")
    plt.plot(tout.T, joint_req_pos_display[6] - joint_actual_pos_display[6], linewidth=3, color="blue",
             label="error value")
    plt.xlabel("time/s")
    plt.ylabel("angle/rad")
    plt.xlim(0, 4)
    plt.legend(loc='best')
    plt.savefig('picture/right/right_joint_e1.eps', format='eps')

    fig8 = plt.figure(8)
    plt.subplot(1, 1, 1)
    plt.title("right_torques")
    plt.plot(tout.T, joint_effort_display[0], linewidth=3, label='joint_w0')
    plt.plot(tout.T, joint_effort_display[1], linewidth=3, label='joint_w1')
    plt.plot(tout.T, joint_effort_display[2], linewidth=3, label='joint_w2')
    plt.plot(tout.T, joint_effort_display[3], linewidth=3, label='joint_e0')
    plt.plot(tout.T, joint_effort_display[4], linewidth=3, label='joint_e1')
    plt.plot(tout.T, joint_effort_display[5], linewidth=3, label='joint_s0')
    plt.plot(tout.T, joint_effort_display[6], linewidth=3, label='joint_s1')
    plt.xlabel("time/s")
    plt.ylabel("torque/Nm")
    plt.xlim(0, 4)
    plt.legend(loc='best', bbox_to_anchor=(1.0,0.7))
    # fig8.savefig('123456.png', transparent=True)
    plt.savefig('picture/right/right_torques.eps', format='eps')

    fig9 = plt.figure(9)
    ax = fig9.add_subplot(111, projection='3d')
    plt.title("right_endpoint_position")
    ax.plot(xyz_display[0], xyz_display[1], xyz_display[2], label="actual value",color='red', linewidth=3)
    ax.plot(xyz_req_display[0], xyz_req_display[1], xyz_req_display[2], label="desired value", color='green', linewidth=3)
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    plt.ylim(-1, 1)
    ax.set_zlim(-1, 1.5)
    plt.legend(loc='best', bbox_to_anchor=(1.1, 1))
    plt.savefig('picture/right/right_endpoint_position.eps', format='eps')

    plt.show()

def main():
    """ 初始化机器人节点和机器人 """
    rospy.init_node("dual_arm_NN")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    rs.enable()

    left_arm = baxter_interface.Limb("left")
    t1 = left_arm.joint_angles()
    t2 = [0.109, 0.823, -1.758, -1.657, 2.060, 0.478, 0.008]  # posi(0.5,0.0,0.4)orien(-0.5,-0.5,-0.5,0.5)

    temp = 0
    for key in t1:
        t1[key] = t2[temp]
        temp = temp + 1
    left_arm.move_to_joint_positions(t1)

    right_arm = baxter_interface.Limb("right")
    t3 = right_arm.joint_angles()
    t4 = [-0.509, -0.095, -3.049, -0.711, -1.613, 1.580, 2.154]   # posi(0.5,0.0,0.4)orien(-0.5,-0.5,-0.5,0.5)
    temp = 0
    for key in t3:
        t3[key] = t4[temp]
        temp = temp + 1
    right_arm.move_to_joint_positions(t3)

    try:
        left_arm_control()
        # thread_l = threading.Thread(target=left_arm_control)
        # thread_r = threading.Thread(target=right_arm_control)
        # thread_r.start()
        # thread_l.start()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()