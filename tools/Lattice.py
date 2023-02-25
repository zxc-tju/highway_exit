#! /usr/bin/env python3
# _*_ coding: utf-8 _*_

import math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.measurements import label
import scipy.signal

M_PI = 3.141593

#车辆属性, global const. and local var. check!
VEH_L = 4  # length
VEH_W = 2  # width
MAX_V = 200
MIN_V = -10000

MAX_A = 10
MIN_A = -20
MAX_LAT_A = 100  #参考apollo，横向约束应该是给到向心加速度，而不是角速度


#cost权重
SPEED_COST_WEIGHT = 1  #速度和目标速度差距，暂时不用
DIST_TRAVEL_COST_WEIGHT = 1  #实际轨迹长度，暂时不用
LAT_COMFORT_COST_WEIGHT = 1  #横向舒适度
LAT_OFFSET_COST_WEIGHT = 1  #横向偏移量

#前四个是中间计算时用到的权重，后三个是最终合并时用到的
LON_OBJECTIVE_COST_WEIGHT = 1  #纵向目标cost，暂时不用
LAT_COST_WEIGHT = 1  #横向约束，包括舒适度和偏移量
LON_COLLISION_COST_WEIGHT = 1  #碰撞cost

def NormalizeAngle(angle_rad):
        # to normalize an angle to [-pi, pi]
        a = math.fmod(angle_rad + M_PI, 2.0 * M_PI)
        if a < 0.0:
            a = a + 2.0 * M_PI
        return a - M_PI

def Dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

class PathPoint:
    def __init__(self, pp_list):
        # pp_list: from CalcRefLine, [rx, ry, rs, rtheta, rkappa, rdkappa] x y 路程 角度 角度变化量/路程变化量 (角度变化量/路程变化量)/路程变化量
        self.rx = pp_list[0]
        self.ry = pp_list[1]
        self.rs = pp_list[2]
        self.rtheta = pp_list[3]
        self.rkappa = pp_list[4]
        self.rdkappa = pp_list[5]

class TrajPoint:
    def __init__(self, tp_list):
        # tp_list: from sensors, [x, y, v, a, theta, kappa]
        self.x = tp_list[0]
        self.y = tp_list[1]
        self.v = tp_list[2]
        self.a = tp_list[3]
        self.theta = tp_list[4]
        self.kappa = tp_list[5]

    def MatchPath(self, path_points):
        '''
        find the closest/projected point on the reference path
        the deviation is not large; the curvature is not large
        '''
        def DistSquare(traj_point, path_point):
            dx = path_point.rx - traj_point.x
            dy = path_point.ry - traj_point.y
            return (dx ** 2 + dy ** 2)
        dist_all = []
        for path_point in path_points:
            dist_all.append(DistSquare(self, path_point))
        dist_min = DistSquare(self, path_points[0])
        index_min = 0
        for index, path_point in enumerate(path_points):
            dist_temp = DistSquare(self, path_point)
            if dist_temp < dist_min:
                dist_min = dist_temp
                index_min = index
        path_point_min = path_points[index_min]
        if index_min == 0 or index_min == len(path_points) - 1:
            self.matched_point = path_point_min
        else:
            path_point_next = path_points[index_min + 1]
            path_point_last = path_points[index_min - 1]
            vec_p2t = np.array([self.x - path_point_min.rx, self.y - path_point_min.ry])
            vec_p2p_next = np.array([path_point_next.rx - path_point_min.rx, path_point_next.ry - path_point_min.ry])
            vec_p2p_last = np.array([path_point_last.rx - path_point_min.rx, path_point_last.ry - path_point_min.ry])
            if np.dot(vec_p2t, vec_p2p_next) * np.dot(vec_p2t, vec_p2p_last) >= 0:
                self.matched_point = path_point_min
            else:
                if np.dot(vec_p2t, vec_p2p_next) >= 0:
                    rs_inter = path_point_min.rs + np.dot(vec_p2t, vec_p2p_next / np.linalg.norm(vec_p2p_next))
                    self.matched_point = LinearInterpolate(path_point_min, path_point_next, rs_inter)
                else:
                    rs_inter = path_point_min.rs - np.dot(vec_p2t, vec_p2p_last / np.linalg.norm(vec_p2p_last))
                    self.matched_point = LinearInterpolate(path_point_last, path_point_min, rs_inter)
        return self.matched_point

    def LimitTheta(self, theta_thr=M_PI/6):
        # limit the deviation of traj_point.theta from the matched path_point.rtheta within theta_thr
        if self.theta - self.matched_point.rtheta > theta_thr:
            self.theta = NormalizeAngle(self.matched_point.rtheta + theta_thr)     # upper limit of theta
        elif self.theta - self.matched_point.rtheta < -theta_thr:
            self.theta = NormalizeAngle(self.matched_point.rtheta - theta_thr)     # lower limit of theta
        else:
            pass    # maintained, actual theta should not deviate from the path rtheta too much

    def IsOnPath(self, dist_thr=0.5):
        # whether the current traj_point is on the path
        dx = self.matched_point.rx - self.x
        dy = self.matched_point.ry - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist <= dist_thr:
            return True
        else:
            return False

#障碍物类
class Obstacle():
    def __init__(self, obstacle_info):
        self.x = obstacle_info[0]
        self.y = obstacle_info[1]
        self.v = obstacle_info[2]
        self.length = obstacle_info[3]
        self.width = obstacle_info[4]
        self.heading = obstacle_info[5]  #这里设定朝向是length的方向，也是v的方向
        self.corner = self.GetCorner()

    def GetCorner(self):
        cos_o = math.cos(self.heading)
        sin_o = math.sin(self.heading)
        dx3 = cos_o * self.length/2
        dy3 = sin_o * self.length/2
        dx4 = sin_o * self.width/2
        dy4 = -cos_o * self.width/2
        return [self.x - (dx3 - dx4), self.y - (dy3 - dy4)]
    
    def MatchPath(self, path_points):
        '''
        find the closest/projected point on the reference path
        the deviation is not large; the curvature is not large
        '''
        def DistSquare(traj_point, path_point):
            dx = path_point.rx - traj_point.x
            dy = path_point.ry - traj_point.y
            return (dx ** 2 + dy ** 2)
        dist_all = []
        for path_point in path_points:
            dist_all.append(DistSquare(self, path_point))  # 求障碍物到reference line的各个点距
        dist_min = DistSquare(self, path_points[0])  # 与第一个参考点的距离
        index_min = 0
        for index, path_point in enumerate(path_points):  # 求最近的参考点
            dist_temp = DistSquare(self, path_point)
            if dist_temp < dist_min:
                dist_min = dist_temp
                index_min = index
        path_point_min = path_points[index_min]  # 得到障碍物到reference line的最短距离
        if index_min == 0 or index_min == len(path_points) - 1:
            self.matched_point = path_point_min
        else:
            path_point_next = path_points[index_min + 1]  # 上一时刻参考点和下一时刻参考点
            path_point_last = path_points[index_min - 1]
            vec_p2t = np.array([self.x - path_point_min.rx, self.y - path_point_min.ry])
            vec_p2p_next = np.array([path_point_next.rx - path_point_min.rx, path_point_next.ry - path_point_min.ry])
            vec_p2p_last = np.array([path_point_last.rx - path_point_min.rx, path_point_last.ry - path_point_min.ry])
            if np.dot(vec_p2t, vec_p2p_next) * np.dot(vec_p2t, vec_p2p_last) >= 0:
                self.matched_point = path_point_min
            else:
                if np.dot(vec_p2t, vec_p2p_next) >= 0:
                    rs_inter = path_point_min.rs + np.dot(vec_p2t, vec_p2p_next / np.linalg.norm(vec_p2p_next))
                    self.matched_point = LinearInterpolate(path_point_min, path_point_next, rs_inter)
                else:
                    rs_inter = path_point_min.rs - np.dot(vec_p2t, vec_p2p_last / np.linalg.norm(vec_p2p_last))
                    self.matched_point = LinearInterpolate(path_point_last, path_point_min, rs_inter)
        return self.matched_point

def CartesianToFrenet(path_point, traj_point):
    ''' from Cartesian to Frenet coordinate, to the matched path point
    copy Apollo cartesian_frenet_conversion.cpp'''
    rx, ry, rs, rtheta, rkappa, rdkappa = path_point.rx, path_point.ry, path_point.rs, \
        path_point.rtheta, path_point.rkappa, path_point.rdkappa
    x, y, v, a, theta, kappa = traj_point.x, traj_point.y, traj_point.v, \
        traj_point.a, traj_point.theta, traj_point.kappa

    s_condition = np.zeros(3)
    d_condition = np.zeros(3)

    dx = x - rx
    dy = y - ry

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)

    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = math.copysign(math.sqrt(dx ** 2 + dy ** 2), cross_rd_nd)

    delta_theta = theta - rtheta
    tan_delta_theta = math.tan(delta_theta)
    cos_delta_theta = math.cos(delta_theta)

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta

    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]

    d_condition[2] = -kappa_r_d_prime * tan_delta_theta + one_minus_kappa_r_d / (cos_delta_theta ** 2) * \
        (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa)

    s_condition[0] = rs
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d

    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_condition[2] = (a * cos_delta_theta - s_condition[1] ** 2 * \
        (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d

    return s_condition, d_condition

def FrenetToCartesian(path_point, s_condition, d_condition):
    ''' from Frenet to Cartesian coordinate
    copy Apollo cartesian_frenet_conversion.cpp'''
    rx, ry, rs, rtheta, rkappa, rdkappa = path_point.rx, path_point.ry, path_point.rs, \
        path_point.rtheta, path_point.rkappa, path_point.rdkappa
    if math.fabs(rs - s_condition[0]) >= 1.0e-6:
        pass
        # print("the reference point s and s_condition[0] don't match")

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)

    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
    delta_theta = math.atan2(d_condition[1], one_minus_kappa_r_d)
    cos_delta_theta = math.cos(delta_theta)
    theta = NormalizeAngle(delta_theta + rtheta)

    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    kappa = ((d_condition[2] + kappa_r_d_prime * tan_delta_theta) * cos_delta_theta ** 2 / one_minus_kappa_r_d \
        + rkappa) * cos_delta_theta / one_minus_kappa_r_d

    d_dot = d_condition[1] * s_condition[1]
    v = math.sqrt((one_minus_kappa_r_d * s_condition[1]) ** 2 + d_dot ** 2)

    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    a = s_condition[2] * one_minus_kappa_r_d / cos_delta_theta + s_condition[1] ** 2 / cos_delta_theta * \
        (d_condition[1] * delta_theta_prime - kappa_r_d_prime)

    tp_list = [x, y, v, a, theta, kappa]
    return TrajPoint(tp_list)

def CalcRefLine(cts_points):  # 输入参考轨迹的x y 计算rs/rtheta/rkappa/rdkappa 此时是笛卡尔坐标系 rs为已走路程 rtheta为角度
    '''
    deal with reference path points 2d-array
    to calculate rs/rtheta/rkappa/rdkappa according to cartesian points
    '''
    rx = cts_points[0]      # the x value
    ry = cts_points[1]      # the y value
    rs = np.zeros_like(rx)
    rtheta = np.zeros_like(rx)
    rkappa = np.zeros_like(rx)
    rdkappa = np.zeros_like(rx)
    for i, x_i in enumerate(rx):
        #y_i = ry[i]
        if i != 0:
            dx = rx[i] - rx[i-1]
            dy = ry[i] - ry[i-1]
            rs[i] = rs[i-1] + math.sqrt(dx ** 2 + dy ** 2)
        if i < len(ry)-1:
            dx = rx[i+1] - rx[i]
            dy = ry[i+1] - ry[i]
            ds = math.sqrt(dx ** 2 + dy ** 2)
            rtheta[i] = math.copysign(math.acos(dx / ds), dy)  # acos求角度 copysign功能为返回第一个输入的值和第二个输入的符号(即dy>0在0-pi dy<0在-pi-0)
    rtheta[-1] = rtheta[-2]  # 最后一个时刻的角度没法求就直接等于倒数第二个时刻
    rkappa[:-1] = np.diff(rtheta) / np.diff(rs)  # 角度变化量/路程变化量
    rdkappa[:-1] = np.diff(rkappa) / np.diff(rs)
    rkappa[-1] = rkappa[-2]
    rdkappa[-1] = rdkappa[-3]
    rdkappa[-2] = rdkappa[-3]
    rkappa = scipy.signal.savgol_filter(rkappa,333,5)  # 平滑
    rdkappa = scipy.signal.savgol_filter(rdkappa,555,5) 
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(rkappa)
    # plt.subplot(212)
    # plt.plot(rdkappa)
    # plt.show()
    path_points = []
    for i in range(len(rx)):
        path_points.append(PathPoint([rx[i], ry[i], rs[i], rtheta[i], rkappa[i], rdkappa[i]]))  # 生成笛卡尔坐标系下的参考轨迹点
    return path_points

def LinearInterpolate(path_point_0, path_point_1, rs_inter):
    ''' path point interpolated linearly according to rs value
    path_point_0 should be prior to path_point_1'''
    def lerp(x0, x1, w):
        return x0 + w * (x1 - x0)
    def slerp(a0, a1, w):
        # angular, for theta
        a0_n = NormalizeAngle(a0)
        a1_n = NormalizeAngle(a1)
        d = a1_n - a0_n
        if d > M_PI:
            d = d - 2 * M_PI
        elif d < -M_PI:
            d = d + 2 * M_PI
        a = a0_n + w * d
        return NormalizeAngle(a)
    rs_0 = path_point_0.rs
    rs_1 = path_point_1.rs
    weight = (rs_inter - rs_0) / (rs_1 - rs_0)
    if weight < 0 or weight > 1:
        print("weight error, not in [0, 1]")
        quit()
    rx_inter = lerp(path_point_0.rx, path_point_1.rx, weight)
    ry_inter = lerp(path_point_0.ry, path_point_1.ry, weight)
    rtheta_inter = slerp(path_point_0.rtheta, path_point_1.rtheta, weight)
    rkappa_inter = lerp(path_point_0.rkappa, path_point_1.rkappa, weight)
    rdkappa_inter = lerp(path_point_0.rdkappa, path_point_1.rdkappa, weight)
    return PathPoint([rx_inter, ry_inter, rs_inter, rtheta_inter, rkappa_inter, rdkappa_inter])

def TrajObsFree(xoy_traj, obstacle, delta_t):  ### 输入为路径点 障碍物类 帧长
    if obstacle.v == 0:
        dis_sum = 0
        for point in xoy_traj:
            if isinstance(point, PathPoint):  # 如果是原来路径点，就只按圆形计算。因为每点的车辆方向难以获得
                ### isinstance 当参数1和参数2是同一类型时返回True
                if ColliTestRough(point, obstacle) > 0:  ### 返回point与obstacle的距离
                    continue
                return 0, False
            else:
                dis = ColliTestRough(point, obstacle)  ### isinstance执行时是路径点与障碍物的距离(是否碰撞) 而else里point不再是路径点(与PathPoint不同类)因此是车辆的位置
                dis_sum += dis
                if dis > 0:
                    continue
                if ColliTest(point, obstacle):  ### 对于车辆与障碍物是否碰撞 ColliTestRough不足以(将两者视为圆形) 要用更准确的ColliTest检测是否碰撞
                    # print("不满足实际碰撞检测")
                    return 0, False
        dis_mean = dis_sum/len(xoy_traj)
        #print("满足实际碰撞检测")
        return dis_mean, True

    if obstacle.v != 0:  ### 如果障碍物速度不为0 要考虑障碍物的位置变化
        dis_sum = 0
        for point in xoy_traj:
            obstacle.x += obstacle.v * delta_t * math.cos(obstacle.heading)
            obstacle.y += obstacle.v * delta_t * math.sin(obstacle.heading)
            if isinstance(point, PathPoint):  #如果是原来路径点，就只按圆形计算。因为每点的车辆方向难以获得
                if ColliTestRough(point,obstacle) > 0:
                    continue
                return 0, False
            else:
                dis = ColliTestRough(point,obstacle)
                dis_sum += dis
                if dis > 0:
                    continue
                if ColliTest(point, obstacle):
                    #print("不满足实际碰撞检测")
                    return 0, False
        dis_mean = dis_sum/len(xoy_traj)
        #print("满足实际碰撞检测")
        return dis_mean, True

#粗略的碰撞检测(视作圆形)  如果此时不碰撞，就无需按矩形检测。返回的距离作为该点车到障碍物的大致距离（无碰撞时也可能为负）
def ColliTestRough(point, obs):
    if isinstance(point, PathPoint):
        dis = math.sqrt((point.rx - obs.x)**2 + (point.ry - obs.y)**2)
    else:
        dis = math.sqrt((point.x - obs.x)**2 + (point.y - obs.y)**2)
    global VEH_L, VEH_W
    max_veh = max(VEH_L, VEH_W)
    max_obs = max(obs.length, obs.width)
    return dis - (max_veh + max_obs)/2

#碰撞检测 (这部分参考apollo代码)
def ColliTest(point, obs):

    shift_x = obs.x - point.x
    shift_y = obs.y - point.y

    global VEH_L, VEH_W
    cos_v = math.cos(point.theta)
    sin_v = math.sin(point.theta)
    cos_o = math.cos(obs.heading)
    sin_o = math.sin(obs.heading)
    half_l_v = VEH_L/2
    half_w_v = VEH_W/2
    half_l_o = obs.length/2
    half_w_o = obs.width/2

    dx1 = cos_v * VEH_L/2
    dy1 = sin_v * VEH_L/2
    dx2 = sin_v * VEH_W/2
    dy2 = -cos_v * VEH_W/2
    dx3 = cos_o * obs.length/2
    dy3 = sin_o * obs.length/2
    dx4 = sin_o * obs.width/2
    dy4 = -cos_o * obs.width/2

    # 使用分离轴定理进行碰撞检测
    return ((abs(shift_x * cos_v + shift_y * sin_v) <=
             abs(dx3 * cos_v + dy3 * sin_v) + abs(dx4 * cos_v + dy4 * sin_v) + half_l_v)
            and (abs(shift_x * sin_v - shift_y * cos_v) <=
                 abs(dx3 * sin_v - dy3 * cos_v) + abs(dx4 * sin_v - dy4 * cos_v) + half_w_v)
            and (abs(shift_x * cos_o + shift_y * sin_o) <=
                 abs(dx1 * cos_o + dy1 * sin_o) + abs(dx2 * cos_o + dy2 * sin_o) + half_l_o)
            and (abs(shift_x * sin_o - shift_y * cos_o) <=
                 abs(dx1 * sin_o - dy1 * cos_o) + abs(dx2 * sin_o - dy2 * cos_o) + half_w_o))

#对符合碰撞和约束限制的轨迹对进行cost排序，目前只保留了碰撞和横向两个cost
def CostSorting(traj_pairs):
    cost_dict = {}
    num = 0
    global LAT_COST_WEIGHT, LON_COLLISION_COST_WEIGHT
    for i in traj_pairs:
        traj = i[0]
        lat_cost = traj.lat_cost  #横向偏移和横向加速度
        lon_collision_cost = -i[1]  #碰撞风险：apollo的较复杂，这里直接用轨迹上各点到障碍物圆的平均距离表示
        cost_dict[num] = lat_cost * LAT_COST_WEIGHT + lon_collision_cost * LON_COLLISION_COST_WEIGHT
        num += 1
    #print(cost_dict)
    cost_list_sorted = sorted(cost_dict.items(), key=lambda d:d[1], reverse=False)
    #print(cost_list_sorted)
    return cost_list_sorted

class PolyTraj:
    def __init__(self, s_cond_init, d_cond_init, total_t):
        self.s_cond_init = s_cond_init
        self.d_cond_init = d_cond_init
        self.total_t = total_t          # to plan how long in seconds
        self.delta_s = 0

    def __QuinticPolyCurve(self, y_cond_init, y_cond_end, x_dur):  ### 五次多项式拟合
        '''
        form the quintic polynomial curve: y(x) = a0 + a1 * delta_x + ... + a5 * delta_x ** 5, x_dur = x_end - x_init
        y_cond = np.array([y, y', y'']), output the coefficients a = np.array([a0, ..., a5])
        '''
        a0 = y_cond_init[0]
        a1 = y_cond_init[1]
        a2 = 1.0 / 2 * y_cond_init[2]

        T = x_dur
        h = y_cond_end[0] - y_cond_init[0]
        v0 = y_cond_init[1]
        v1 = y_cond_end[1]
        acc0 = y_cond_init[2]
        acc1 = y_cond_end[2]
        #print(x_dur)
        a3 = 1.0 / (2 * T ** 3) * (20 * h - (8 * v1 + 12 * v0) * T - (3 * acc0 - acc1) * T ** 2)
        a4 = 1.0 / (2 * T ** 4) * (-30 * h + (14 * v1 + 16 * v0) * T + (3 * acc0 - 2 * acc1) * T ** 2)
        a5 = 1.0 / (2 * T ** 5) * (12 * h - 6 * (v1 + v0) * T + (acc1 - acc0) * T ** 2)
        return np.array([a0, a1, a2, a3, a4, a5])

    def GenLongTraj(self, s_cond_end):
        self.long_coef = self.__QuinticPolyCurve(self.s_cond_init, s_cond_end, self.total_t)  ### self.long_coef为五次多项式的参数
        self.delta_s = self.long_coef[1] * self.total_t + self.long_coef[2] * self.total_t ** 2 + \
            self.long_coef[3] * self.total_t ** 3 + self.long_coef[4] * self.total_t ** 4 + \
                self.long_coef[5] * self.total_t ** 5
        # return self.long_coef

    def GenLatTraj(self, d_cond_end):
        # GenLatTraj should be posterior to GenLongTraj
        self.lat_coef = self.__QuinticPolyCurve(self.d_cond_init, d_cond_end, self.delta_s)
        # return self.lat_coef

    #求各阶导数
    def Evaluate(self, coef, order, t):
        if order == 0:
            return ((((coef[5] * t + coef[4]) * t + coef[3]) * t
                     + coef[2]) * t + coef[1]) * t + coef[0]
        if order == 1:
            return (((5 * coef[5] * t + 4 * coef[4]) * t + 3 *
                     coef[3]) * t + 2 * coef[2]) * t + coef[1]
        if order == 2:
            return (((20 * coef[5] * t + 12 * coef[4]) * t)
                    + 6 * coef[3]) * t + 2 * coef[2]
        if order == 3:
            return (60 * coef[5] * t + 24 * coef[4]) * t + 6 * coef[3]
        if order == 4:
            return 120 * coef[5] * t + 24 * coef[4]
        if order == 5:
            return 120 * coef[5]

    #纵向速度&加速度约束
    def LongConsFree(self, delta_t):
        size = int(self.total_t/delta_t)
        global MAX_V, MIN_V
        for i in range(size):
            v = self.Evaluate(self.long_coef, 1, i * delta_t)
            #print(v)
            if v > MAX_V or v < MIN_V:
                #print(v, "纵向速度超出约束")
                return False
            '''
            加速度约束暂时删去
            a = self.Evaluate(self.long_coef,2, i*delta_t)
            if a > MAX_A or a < MIN_A:
                print("纵向加速度超出约束")
                return False
            '''
        #print("满足纵向约束")
        return True

    #横向加速度约束，参考apollo。这里把横向的cost一块算了
    #横向偏移量和横向加速度cost同样参考apollo，数学上做了一些简化，如省略了偏移量绝对值，只计算平方；忽略和起点之间的偏移量关系等
    def LatConsFree(self, delta_t):
        size = int(self.total_t/delta_t)
        lat_offset_cost = 0
        lat_comfort_cost = 0
        global LAT_COMFORT_COST_WEIGHT, LAT_OFFSET_COST_WEIGHT
        for i in range(size):
            s = self.Evaluate(self.long_coef, 0, i*delta_t)
            d = self.Evaluate(self.lat_coef, 0, s)
            dd_ds = self.Evaluate(self.lat_coef, 1, s)
            ds_dt = self.Evaluate(self.long_coef, 1, i*delta_t)
            d2d_ds2 = self.Evaluate(self.lat_coef, 2, s)
            d2s_dt2 = self.Evaluate(self.long_coef, 2, i*delta_t)

            lat_a = d2d_ds2 * ds_dt * ds_dt + dd_ds * d2s_dt2
            '''
            向心加速度暂时删去
            if abs(lat_a) > MAX_LAT_A:
                print(lat_a, "不满足横向约束")
                return False
            '''
            lat_comfort_cost += lat_a * lat_a
            lat_offset_cost += d * d

        self.lat_cost = lat_comfort_cost * LAT_COMFORT_COST_WEIGHT + lat_offset_cost * LAT_OFFSET_COST_WEIGHT
        #print("满足横向约束")
        return True

    def GenCombinedTraj(self, path_points, delta_t):
        '''
        combine long and lat traj together
        F2C function is used to output future traj points in a list to follow
        '''
        a0_s, a1_s, a2_s, a3_s, a4_s, a5_s = self.long_coef[0], self.long_coef[1], self.long_coef[2], \
            self.long_coef[3], self.long_coef[4], self.long_coef[5]
        a0_d, a1_d, a2_d, a3_d, a4_d, a5_d = self.lat_coef[0], self.lat_coef[1], self.lat_coef[2], \
            self.lat_coef[3], self.lat_coef[4], self.lat_coef[5]

        rs_pp_all = []              # the rs value of all the path points
        for path_point in path_points:
            rs_pp_all.append(path_point.rs)
        rs_pp_all = np.array(rs_pp_all)
        num_points = math.floor(self.total_t / delta_t)  ### 规划时长/帧长 = 规划点数
        s_cond_all = []             # possibly useless
        d_cond_all = []             # possibly useless
        pp_inter = []               # possibly useless
        tp_all = []                 # all the future traj points in a list
        t, s = 0, 0                 # initialize variables, s(t), d(s) or l(s)
        for i in range(int(num_points)):
            s_cond = np.zeros(3)
            d_cond = np.zeros(3)

            t = t + delta_t
            s_cond[0] = a0_s + a1_s * t + a2_s * t ** 2 + a3_s * t ** 3 + a4_s * t ** 4 + a5_s * t ** 5  # 路程
            s_cond[1] = a1_s + 2 * a2_s * t + 3 * a3_s * t ** 2 + 4 * a4_s * t ** 3 + 5 * a5_s * t ** 4  # 速度(d路程/dt)
            s_cond[2] = 2 * a2_s + 6 * a3_s * t + 12 * a4_s * t ** 2 + 20 * a5_s * t ** 3  # a
            s_cond_all.append(s_cond)

            s = s_cond[0] - a0_s
            d_cond[0] = a0_d + a1_d * s + a2_d * s ** 2 + a3_d * s ** 3 + a4_d * s ** 4 + a5_d * s ** 5
            d_cond[1] = a1_d + 2 * a2_d * s + 3 * a3_d * s ** 2 + 4 * a4_d * s ** 3 + 5 * a5_d * s ** 4
            d_cond[2] = 2 * a2_d + 6 * a3_d * s + 12 * a4_d * s ** 2 + 20 * a5_d * s ** 3
            d_cond_all.append(d_cond)

            index_min = np.argmin(np.abs(rs_pp_all - s_cond[0]))
            path_point_min = path_points[index_min]  ### 现在到哪个位置了
            if index_min == 0 or index_min == len(path_points) - 1:
                path_point_inter = path_point_min
            else:
                if s_cond[0] >= path_point_min.rs:
                    path_point_next = path_points[index_min + 1]
                    path_point_inter = LinearInterpolate(path_point_min, path_point_next, s_cond[0])
                else:
                    path_point_last = path_points[index_min - 1]
                    path_point_inter = LinearInterpolate(path_point_last, path_point_min, s_cond[0])
            pp_inter.append(path_point_inter)
            traj_point = FrenetToCartesian(path_point_inter, s_cond, d_cond)
            #traj_point.v = v_tgt
            tp_all.append(traj_point)
        self.tp_all = tp_all
        return tp_all

class SampleBasis:
    # the basis of sampling: theta, dist, d_end (, v_end); normally for the planning_out cruising case
    def __init__(self, traj_point, theta_thr, ttcs):
        global v_tgt  ### 目标速度
        traj_point.LimitTheta(theta_thr)
        self.theta_samp = [NormalizeAngle(traj_point.theta - theta_thr), NormalizeAngle(traj_point.theta - theta_thr/2), 
                           traj_point.theta, NormalizeAngle(traj_point.theta + theta_thr/2), NormalizeAngle(traj_point.theta + theta_thr)]
                          ### NormalizeAngle将角度转化为[-pi,pi] 角度的采样区间为原轨迹点theta下[-theta_thr,-theta_thr/2,0,theta_thr/2,theta_thr]即最大转向角为theta_thr
        self.dist_samp = [v_tgt * ttc for ttc in ttcs]  ### 距离的采样区间为目标速度*ttcs区间(这里为3s 4s 5s)
        self.dist_prvw = self.dist_samp[0]  # 最小的距离采样
        self.d_end_samp = [0]
        self.v_end = v_tgt      # for cruising

class LocalPlanner:
    def __init__(self, traj_point, path_points, obstacles, samp_basis, aggressiveness):
        self.traj_point_theta = traj_point.theta    # record the current heading
        self.traj_point = traj_point
        self.path_points = path_points
        self.obstacles = obstacles
        self.aggressiveness = aggressiveness
        self.theta_samp = samp_basis.theta_samp
        self.dist_samp = samp_basis.dist_samp
        self.d_end_samp = samp_basis.d_end_samp
        self.v_end = samp_basis.v_end
        self.polytrajs = []
        self.__JudgeStatus(traj_point, path_points, obstacles, samp_basis)

    
    def __JudgeStatus(self,traj_point, path_points, obstacles, samp_basis):  ### 赋值self.status和self.dist_prvw和self.to_Stop# 分别表示车辆的位置关系 最小的采样距离
        colli = 0
        global delta_t, sight_range  ### 每帧时长 可视距离(sight_range下有无障碍物)
        path_point_end = self.path_points[-1]

        # 假设末端也不停车目前
        # if path_point_end.rs - self.traj_point.matched_point.rs <= samp_basis.dist_prvw:  ### 如果快到参考轨迹的末端了(小于最小采样空间的距离) 则准备停车
        #     self.to_stop = True  # stopping
        #     self.dist_prvw = path_point_end.rs - traj_point.matched_point.rs  ### 最小距离赋值
        # else:
        #     self.to_stop = False  # cruising
        self.to_stop = False
        self.dist_prvw = samp_basis.dist_prvw * 2  # make acc near 1m/s2
        colli_match = None

        for obs in range(int(0.5*len(self.obstacles))):
            obstacle = self.obstacles[obs]
            if self.traj_point.matched_point.ry - obstacle.matched_point.ry > 3:
                continue
            if self.traj_point.matched_point.rs - obstacle.matched_point.rs > VEH_L:  ### 障碍物的match_point小于车辆当前match_point
                continue
            if Dist(obstacle.x, obstacle.y, self.traj_point.x, self.traj_point.y) > sight_range:  ### 距离大于可视距离
                # 只看眼前一段距离
                continue
            temp = TrajObsFree(self.path_points, obstacle, delta_t)
            if not temp[1]:  ### 有碰撞 指的是障碍物与参考轨迹是否有碰撞(即是否重合)
                colli = 1
                colli_match = obstacle
                break

        # if self.obstacles != []:
        #     real_obs_pos = self.obstacles[0]  # 实际对手车的位置
        #     if self.traj_point.matched_point.rx - real_obs_pos.matched_point.rx > 0:  # 对手在本车后面 且领先很多
        #         preceded = True
        #     else:
        #         preceded = False
        #         # colli_match_point = real_obs_pos.matched_point
        # else:
        #     preceded = True

        # if colli== 0 or preceded == True:
        if colli == 0:
            self.status = "following_path"  ### 离得远且没有碰撞即planning_back
        else:
            colli_match_point = colli_match.matched_point
            self.dist_prvw = colli_match_point.rs - traj_point.matched_point.rs
            self.status = "following_path"  ### 碰撞即planning_out
            self.to_stop = True
            # if colli == 0:
            #     if traj_point.IsOnPath():  ### 车辆现在在不在参考轨迹上(标准为traj的match_point与当前traj的距离是否小于0.5m)
            #         self.status = "following_path"
            #     else:
            #         self.status = "planning_back"  ### 离得远且没有碰撞即planning_back
            # else:
            #     self.status = "planning_out"  ### 碰撞即planning_out


    def __LatticePlanner(self,traj_point, path_points):
        # core algorithm of the lattice planner

        # plt.figure()
        # plt.plot(rx, ry, 'b')
        # for obstacle in self.obstacles:
        #     plt.gca().add_patch(plt.Rectangle((obstacle.corner[0], obstacle.corner[1]), obstacle.length,
        #                  obstacle.width, color='r', angle = obstacle.heading*180/M_PI))

        global delta_t, v_tgt, sight_range
        colli_free_traj_pairs = []                      # PolyTraj object with corresponding trajectory's cost
        for theta in self.theta_samp:                   # theta (heading) samping  ### 航向角采样空间
            self.traj_point.theta = theta
            s_cond_init, d_cond_init = CartesianToFrenet(self.traj_point.matched_point, self.traj_point)  ### 转化坐标系 s d分别速度方向和垂直于速度方向
            s_cond_init[2], d_cond_init[2] = 0, 0  # [0]为该坐标系下路程 [1]为速度
            #if  s_cond_init[1] > 1 * v_tgt:
            #s_cond_init[1] = self.traj_point.v
            #print("aa", s_cond_init, d_cond_init)
            #print(self.dist_samp)
            for delta_s in self.dist_samp:              # s_cond_end[0] sampling
                total_t = delta_s / v_tgt
                poly_traj = PolyTraj(s_cond_init, d_cond_init, total_t)
                s_cond_end = np.array([s_cond_init[0] + delta_s, self.v_end, 0])  ### v_end = v_tgt
                poly_traj.GenLongTraj(s_cond_end)  ### GenLongTraj GenLatTraj分别得到五次多项式的系数
                if not poly_traj.LongConsFree(delta_t):  # 先看纵向轨迹s是否满足纵向运动约束
                    pass
                else:
                    for d_end in self.d_end_samp:       # d_end[0] sampling self.d_end_samp = [0]
                        d_cond_end = np.array([d_end, 0, 0])
                        poly_traj.GenLatTraj(d_cond_end)
                        tp_all = poly_traj.GenCombinedTraj(self.path_points, delta_t)  ### 生成规划轨迹 tp_all是笛卡尔坐标系
                        self.polytrajs.append(poly_traj)
                        colli = 0
                        dis_to_obs = 0
                        for obstacle in self.obstacles:
                            if obstacle.matched_point.rs < self.traj_point.matched_point.rs:
                                continue
                            if Dist(obstacle.x, obstacle.y, traj_point.x, traj_point.y) > sight_range:
                                #只看眼前一段距离
                                continue
                            temp = TrajObsFree(tp_all, obstacle, delta_t)  ### 返回距离 是否碰撞
                            if not temp[1]:   #有碰撞
                                colli = 1
                                break
                            dis_to_obs += temp[0]
                        if colli == 0:
                            if poly_traj.LatConsFree(delta_t):  # 满足横向约束
                                #print("available trajectory found")
                                colli_free_traj_pairs.append([poly_traj, dis_to_obs])  ### 把在theta delta_s采样下的轨迹收集 并且同时保留与障碍物的距离dis_to_obs
                            tp_x, tp_y, tp_v, tp_a = [], [], [], []
                            for tp in tp_all:
                                tp_x.append(tp.x)
                                tp_y.append(tp.y)
                                tp_v.append(tp.v)
                                tp_a.append(tp.a)

                            # plt.figure()
                            # plt.plot(tp_v)
                            # plt.plot(tp_x, tp_y, 'k')
                            # plt.plot(self.traj_point.x, self.traj_point.y, 'or')
                            # plt.axis('scaled')
                            # plt.xlim(-270,-250)
                            # plt.ylim(-504,-484)
        


        if colli_free_traj_pairs:      # selecting the best one
            cost_list = CostSorting(colli_free_traj_pairs)  ### 得到一个traj_pairs(规划轨迹)的升序损失函数序列
            cost_min_traj = colli_free_traj_pairs[cost_list[0][0]][0]
            traj_points_opt = cost_min_traj.tp_all
            tpo_x = []
            tpo_y = []
            for tpo in traj_points_opt:
                tpo_x.append(tpo.x)
                tpo_y.append(tpo.y)
            # plt.plot(tpo_x, tpo_y, '.g')
            # plt.show()
            return traj_points_opt  ### 从一堆无碰撞轨迹中得到一个损失最小的可行轨迹(笛卡尔坐标系下)
        else:                       # emergency stop
            # print("没找到可行解, emergency stop needed")
            return False

    def __PathFollower(self, traj_point, path_points, target_v):  #无障碍物且在原轨迹上时的循迹,认为从matched_point开始
        #匀变速运动，使速度满足在到达预瞄距离时等于v_end，即v_tgt(未到终点)或0(到终点)
        # plt.figure()
        # plt.plot(rx, ry, 'b')
        # plt.plot(self.traj_point.x, self.traj_point.y, 'or')
        # for obstacle in self.obstacles:
        #     plt.gca().add_patch(plt.Rectangle((obstacle.corner[0], obstacle.corner[1]), obstacle.length,
        #                  obstacle.width, color='r', angle = obstacle.heading*180/M_PI))

        global delta_t
        if target_v == 0:  # 360m内减速停下 v²/2a
            self.v_end = 0
            self.dist_prvw = 400  # 使减速度在1m/s2左右
        elif target_v == v_tgt:
            pass
        else:  # 20m内到达规划速度（!= v_tgt且!=0 只可能出现在同车道前车稍慢于本车当前速度）
            self.v_end = target_v
            self.dist_prvw = 50
        acc = ((self.v_end ** 2 - self.traj_point.v ** 2) / (2 * self.dist_prvw) + 10e-10)  ### 以此加速度向v_end靠拢(在to_stop=False时=v_tgt)
        total_t = 2 * self.dist_prvw / (self.v_end + self.traj_point.v)
        num_points = math.floor(total_t / delta_t)
        tp_all = []                 # all the future traj points in a list
        rs_pp_all = []              # the rs value of all the path points
        tp_x = []
        tp_y = []
        for path_point in path_points:
            rs_pp_all.append(path_point.rs)
        rs_pp_all = np.array(rs_pp_all)
        for i in range(int(num_points)):
            s_cond = np.zeros(3)
            d_cond = np.zeros(3)
            s_cond[0] = (self.traj_point.matched_point.rs 
                         + self.traj_point.v * i * delta_t + (1/2) * acc * ((i * delta_t) ** 2))
            s_cond[1] = self.traj_point.v + acc * i * delta_t
            s_cond[2] = acc  ### 此时的路程 速度 加速度 由于沿着参考轨迹行驶 所以在frenet下没有横向数据
            index_min = np.argmin(np.abs(rs_pp_all - s_cond[0]))
            path_point_min = path_points[index_min]
            if index_min == 0 or index_min == len(path_points) - 1:
                path_point_inter = path_point_min
            else:
                if s_cond[0] >= path_point_min.rs:
                    path_point_next = path_points[index_min + 1]
                    path_point_inter = LinearInterpolate(path_point_min, path_point_next, s_cond[0])
                else:
                    path_point_last = path_points[index_min - 1]
                    path_point_inter = LinearInterpolate(path_point_last, path_point_min, s_cond[0])

            traj_point = FrenetToCartesian(path_point_inter, s_cond, d_cond)
            tp_all.append(traj_point)
            tp_x.append(traj_point.x)
            tp_y.append(traj_point.y)

        if len(tp_all) >= 2:  # 小于2连next_traj_point = tp_all[1]都没有
            dis_to_obs = 0

            for obstacle in self.obstacles:
                if self.traj_point.matched_point.rs - obstacle.matched_point.rs > VEH_L:
                    continue
                if Dist(obstacle.x, obstacle.y, traj_point.x, traj_point.y) > sight_range:
                    # 只看眼前一段距离
                    continue
                # plt.gca().add_patch(plt.Rectangle((obstacle.corner[0], obstacle.corner[1]), obstacle.length, obstacle.width, color='y', angle = obstacle.heading*180/M_PI))
                temp = TrajObsFree(tp_all, obstacle, delta_t)  ### 返回距离 是否碰撞
                dis_to_obs += temp[0]

            r = dis_to_obs
            # plt.plot(tp_x, tp_y, '.g')
            # plt.axis('scaled')
            # plt.xlim(-270,-250)
            # plt.ylim(-504,-484)
            # plt.show()
            # tunring
            if self.obstacles != []:
                next_traj_point = tp_all[1]
                p_self_now = [next_traj_point.x, next_traj_point.y, next_traj_point.v, next_traj_point.theta]
                p_self_last = [self.traj_point.x, self.traj_point.y]
                p_opp_last = [self.obstacles[1].x, self.obstacles[1].y]
                mo = math.sqrt(
                    (p_opp_last[1] - p_self_last[1]) ** 2 + (p_opp_last[0] - p_self_last[0]) ** 2)
                u = ((p_opp_last[1] - p_self_last[1]) * (p_self_last[1] - p_self_now[1]) + (
                        p_opp_last[0] - p_self_last[0]) * (p_self_last[0] - p_self_now[0])) / mo

            else:
                u = 0

            weight = 50
            r = r + u * self.aggressiveness * weight
            # print(u * self.aggressiveness * weight)
            return tp_all, r  ### 同样转化回笛卡尔坐标系
        else:
            return [], -10000000

# 横向加速度阈值

    def __FollowingPath(self,traj_point, path_points, ref_changed):
        if self.to_stop:
            self.v_end = 0  ### else v_end = v_tgt
        return self.__PathFollower(traj_point, path_points, ref_changed)

    def __PlanningOut(self,traj_point, path_points):
        if self.to_stop:    # stopping
            #self.dist_samp = [self.dist_prvw]
            self.v_end = 0
        return self.__LatticePlanner(traj_point, path_points)  ### 规划出损失最小的轨迹

    def __PlanningBack(self,traj_point, path_points):
        self.theta_samp = [self.traj_point_theta]   # just use the current heading, is it necessary?
        self.dist_samp = [self.dist_prvw]           # come back asap, is it necessary?
        self.d_end_samp = [0]
        if self.to_stop:    # stopping
            self.v_end = 0
        return self.__LatticePlanner(traj_point, path_points)

    def LocalPlanning(self,traj_point, path_points, target_v):
        if self.status == "following_path":  ### 在参考轨迹上且无碰撞的风险
            return self.__FollowingPath(traj_point, path_points, target_v)
        elif self.status == "planning_out":  ### 与障碍物有冲突 需要离开参考轨迹
            return self.__PlanningOut(traj_point, path_points)
        elif self.status == "planning_back":  ### 无冲突且不在参考轨迹上 要回到参考轨迹
            return self.__PlanningBack(traj_point, path_points)
        else:
            quit()

def Moving_obstacle(obstacle_info, past_time, total_t=2):
    num_points = total_t/delta_t
    obstacles = []
    for point in range(int(num_points)):
        point += 1
        obstacle_x = obstacle_info.x + obstacle_info.v * delta_t * math.cos(obstacle_info.heading) * (point + past_time)
        obstacle_y = obstacle_info.y + obstacle_info.v * delta_t * math.sin(obstacle_info.heading) * (point + past_time)
        obstacles.append(Obstacle([obstacle_x, obstacle_y, obstacle_info.v, obstacle_info.length, obstacle_info.width, obstacle_info.heading]))
    return obstacles

def sigmoid_function(pos, steepness):  # diff # sigmoid函数的陡峭程度 越小越平缓
    lane_width = 4  # 车道长(换道需要的横向距离)
    lane_changing_length = 100  # 换道需要的纵向距离
    x = pos[0]
    y = pos[1]
    z = np.arange(x, x+lane_changing_length, ref_line_interval)
    fz = np.zeros((0, 2))
    for num in z:
        sig = 1 / (1 + math.exp(-steepness*(num - x - lane_changing_length/2)))  # sigmoid函数的中点是x+lane_changing_length/2
        fz = np.vstack((fz, [num, -sig * lane_width + y]))

    max_y = max(fz[:, 1])
    min_y = min(fz[:, 1])
    gap = abs(max_y - min_y)
    for y in range(len(z)):
        fz[y, 1] = (fz[y, 1] - min_y) * lane_width / gap + int(min_y/4) * 4
    return fz

def straight_line(pos):
    x = pos[0]
    y = pos[1]
    lane_changing_length = 100  # 换道需要的纵向距离
    z = np.arange(x, x + lane_changing_length, ref_line_interval)
    fz = np.zeros((0, 2))
    for num in z:
        fz = np.vstack((fz, [num, y]))
    return fz

'''
# the test example from Apollo conversion_test.cpp
rs = 10.0
rx = 0.0
ry = 0.0
rtheta = M_PI / 4.0
rkappa = 0.1
rdkappa = 0.01
x = -1.0
y = 1.0
v = 2.0
a = 0.0
theta = M_PI / 3.0
kappa = 0.11
pp_list = [rx, ry, rs, rtheta, rkappa, rdkappa]
tp_list = [x, y, v, a, theta, kappa]
pp_in = PathPoint(pp_list)
tp_in = TrajPoint(tp_list)
s_cond, d_cond = CartesianToFrenet(pp_in, tp_in)
tp_rst = FrenetToCartesian(pp_in, s_cond, d_cond)
print(tp_rst.x, tp_rst.y, tp_rst.v, tp_rst.a, tp_rst.theta, tp_rst.kappa)
'''
steep_list = []
ref_line_interval = 0.05
delta_t = 0.1 * 1                           # fixed time between two consecutive trajectory points, sec
v_tgt = 100/3.6                                # fixed target speed, m/s
sight_range = 50                            # 判断有无障碍物的视野距离
destination = [100, 2]
theta_thr = M_PI / 12  # delta theta threshold, deviation from matched path
ttcs = [3]  # static ascending time-to-collision, sec

# ref1 = sigmoid_function([0, 8], steepness=0.1)
# ref2 = sigmoid_function([50, 0], steepness=0.05)
# plt.figure()
# plt.plot(ref1[:,0], ref1[:,1])
# plt.plot(ref2[:,0], ref2[:,1])
# plt.show()

# if __name__ == '__main__':
#     time1 = time.time()
#     # global: delta_t, v_tgt
#     # from listener: path_data, tp_list, obstacles
#     # sampling: input theta_thr, ttcs; ouput theta_samp, dist_samp, d_end_samp
#     # to talker: traj_points (list) or False
#     path_data = sigmoid_function([0, 10])  ### 参考轨迹数据集合 只需要x y
#     rx = path_data[:, 0]
#     ry = path_data[:, 1]
#     cts_points = np.array([rx, ry])
#     path_points = CalcRefLine(cts_points)  ### 根据路径.txt文件中的x y 生成笛卡尔坐标系类
#     theta_thr = M_PI/6                          # delta theta threshold, deviation from matched path
#     ttcs = [3, 4, 5]                            # static ascending time-to-collision, sec
#     # dist_samp = [v_tgt * ttc for ttc in ttcs]   # sampling distances, m
#     # dist_prvw = dist_samp[0]                    # preview distance, equal to minimal sampling distance
#     # d_end_samp = [0]                            # sampling d_cond_end[0], probably useless
#     obstacles = []
#     obs1 = Obstacle([0, 6, 2, 4, 2, 0 * M_PI])
#     obstacles.append(obs1)  ### 障碍参数包括x y v 长 宽 方向(与长同向)
#     # obstacles.append(Obstacle([rx[300]+1, ry[300], 0, 1, 1, M_PI/2]))
#     # obstacles.append(Obstacle([rx[500]+1, ry[500], 0, 1, 1, M_PI/3]))
#     # theta_init = math.atan2((ry[1]-ry[0]), (rx[1]-rx[0]))
#     tp_list = [rx[0], ry[0], 0, 0, 3., 0]   # from sensor actually, an example here
#     traj_point = TrajPoint(tp_list)  ### [x, y, v, a, theta, kappa]
#     traj_point.MatchPath(path_points)           # matching once is enough 将traj_point(单点)与path_points(序列)中最近的点匹配
#     for obstacle in obstacles:
#         obstacle.MatchPath(path_points)  ### 同样match障碍物与参考轨迹
#     # traj_point.LimitTheta(theta_thr)            # limit the deviation
#     # theta_samp = [traj_point.theta - theta_thr, traj_point.theta, traj_point.theta + theta_thr]
#
#     samp_basis = SampleBasis(traj_point, theta_thr, ttcs)  ### 采样区间(类动作空间)
#     local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis)  ### 规划器 输入为目前位置 参考轨迹点 障碍物位置 采样空间
#     print(local_planner.status, local_planner.to_stop)
#     ### 规划器生成时已经可以根据初状态确定车辆的status和to_stop status是车辆是否在参考轨迹上以及是否碰撞 to_stop是车辆是巡航还是准备停车(这里的准备停车仅指要到轨迹末端了所以停车)
#     traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, obstacles, samp_basis)  ### 规划器根据不同的status进行规划
#     print(local_planner.v_end, samp_basis.v_end)
#
#     v_list = []
#     a_list = []
#
#
#     # plt.figure()
#
#     i = 0
#     while(Dist(traj_point.x, traj_point.y, rx[-1], ry[-1]) > 2):  ### while traj还没到终点
#         i += 1
#         traj_point = traj_points_opt[1]  ### 规划轨迹的点 此时的traj_points_opt是进循环前的0时刻 也就是i=0 这也是为什么要先i+=1 [1]是表示现在位置是上一时刻规划的第二个点(第一个点是上一时刻的位置)
#         traj_point.MatchPath(path_points)
#
#         print(Dist(traj_point.x, traj_point.y, rx[-1], ry[-1]), traj_point.y - destination[1])
#         if (Dist(traj_point.x, traj_point.y, rx[-1], ry[-1]) < 4) and traj_point.y - destination[1] > 1:  # 连续变道
#             path_data = sigmoid_function([traj_point.x, traj_point.y])  ### 参考轨迹数据集合 只需要x y
#             rx = path_data[:, 0]
#             ry = path_data[:, 1]
#             cts_points = np.array([rx, ry])
#             path_points = CalcRefLine(cts_points)
#             traj_point.MatchPath(path_points)
#
#         obstacles = []
#         obstacles = np.hstack((obstacles, Moving_obstacle(obs1, i)))  ### 障碍参数包括x y v 长 宽 方向(与长同向)
#         for obstacle in obstacles:
#             obstacle.MatchPath(path_points)  ### 同样match障碍物与参考轨迹
#
#         samp_basis = SampleBasis(traj_point, theta_thr, ttcs)
#         local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis)
#         traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, obstacles, samp_basis)
#         #如果采样较少的情况未找到可行解，考虑扩大采样范围
#         if not traj_points_opt:
#             print("扩大范围")
#             theta_thr_ = M_PI/3
#             ttcs_ = [2, 3, 4, 5, 6, 7, 8]
#             samp_basis = SampleBasis(traj_point, theta_thr_, ttcs_)
#             local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis)
#             traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, obstacles, samp_basis)
#         if not traj_points_opt:
#             traj_points = [[0,0,0,0,0,0]]
#             print("无解")
#             break
#         else:  ### 正常情况下在正常采样空间内如果有opt 就将opt规划出的点作为下一时刻的traj
#             traj_points=[]
#             for tp_opt in traj_points_opt:
#                 traj_points.append([tp_opt.x,tp_opt.y,tp_opt.v,tp_opt.a,tp_opt.theta,tp_opt.kappa])
#         ### traj这里就是用于画图 位置变化即从当前位置(traj_points_opt[0])到下一时刻位置(traj_points_opt[1])
#         tx = [x[0] for x in traj_points]
#         ty = [y[1] for y in traj_points]
#         plt.ion()
#         plt.figure(1)
#         # plt.xlim(0, 100)
#         # plt.ylim(-1, 13)
#         plt.plot(np.arange(0, 100, 1), 0 * np.ones(100), color='grey', linestyle='dotted')
#         plt.plot(np.arange(0, 100, 1), 4 * np.ones(100), color='grey', linestyle='dotted')
#         plt.plot(np.arange(0, 100, 1), 8 * np.ones(100), color='grey', linestyle='dotted')
#         plt.plot(np.arange(0, 100, 1), 12 * np.ones(100), color='grey', linestyle='dotted')
#         plt.plot(rx, ry, 'b')  ### rx ry是参考轨迹
#         plt.plot(traj_point.x, traj_point.y, 'or')  ### 当前位置
#         plt.plot(tx, ty,'r')  ### tx ty是规划轨迹
#         for obstacle in obstacles:  ### 画障碍物
#             plt.gca().add_patch(plt.Rectangle((obstacle.corner[0], obstacle.corner[1]), obstacle.length,
#                         obstacle.width, color='r', angle=obstacle.heading*180/M_PI))
#             plt.axis('scaled')
#         plt.pause(0.0002)
#         plt.clf()
#         """plt.close()
#         print(traj_point.v)
#         v_list.append(traj_point.v)
#         a_list.append(traj_point.a)
#         print(local_planner.status, local_planner.to_stop)
#         v_tra = [a[2] for a in traj_points ]
#         a_tra = [b[3] for b in traj_points ]
#         fig1,ax1 = plt.subplots()
#         ax2 = plt.twinx(ax1)
#         ax1.plot(v_tra,'b')
#         ax2.plot(a_tra,'r')
#         ax1.set_xlabel('time/s')
#         ax1.set_ylabel('v',color='b')
#         ax2.set_ylabel('a',color='r')
#         plt.show()
#         plt.pause(0.0002)
#         plt.clf() """
#     time2 = time.time()
#     print(time2-time1)
