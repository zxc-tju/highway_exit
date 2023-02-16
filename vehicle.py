import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as mt
from agent import Agent
from tools.utility import smooth_ployline
from tools.Lattice import *

dt = 0.1


class Vehicle:
    def __init__(self, source_lane, road_net):
        self.is_av = False
        self.delta_speed_from_front = -999
        self.distance_from_front = 999
        self.global_id = road_net.vehicle_number
        self.width = 2
        self.length = 4
        self.lane_id = source_lane.id
        self.last_lane_id = None
        self.rank_in_lane = source_lane.vehicle_number
        self.current_cv = None

        # sample reaction feature
        self.react_threshold = np.random.uniform(0, 3.5)

        # sample initial position
        self.x = np.random.uniform(-1, 5)
        self.y = np.mean(source_lane.y_range)
        self.heading = 0

        # sample initial motion and IDM parameters
        self.vx = (100 + np.random.uniform(-10, 10)) / 3.6
        self.vy = 0

        self.front_vehicle = None
        self.source_lane = source_lane
        self.get_leading_vehicle_info()

        self.idm_parameter = [
            2,  # minimum spacing at a standstill (m)
            2 + np.random.uniform(-0.5, 0.5),  # desired time headway (s)
            5,  # max acceleration (m/s^2)
            2 + np.random.uniform(-1, 1),  # comfortable acceleration (m/s^2)
            (120 + np.random.uniform(-10, 10)) / 3.6  # desired speed (m/s)
        ]
        road_net.vehicle_number += 1
        source_lane.vehicle_number += 1

    def get_leading_vehicle_info(self, given_leading_car=None):
        self.front_vehicle = None
        if given_leading_car:
            self.front_vehicle = given_leading_car
        else:
            for veh in self.source_lane.vehicle_list:
                if veh.rank_in_lane == self.rank_in_lane - 1:
                    self.front_vehicle = veh

        if self.front_vehicle:
            self.distance_from_front = self.front_vehicle.x - self.x
            self.delta_speed_from_front = self.vx - self.front_vehicle.vx
        else:
            self.delta_speed_from_front = -999
            self.distance_from_front = 999

    def get_central_vertices(self):
        cv_current_x = self.source_lane.x_range
        cv_current_y = np.mean(self.source_lane.y_range)
        cv_current_start = [cv_current_x[0], cv_current_y]
        cv_current_end = [cv_current_x[1], cv_current_y]
        current_cv_raw = np.array([cv_current_start,
                                   [cv_current_x[0] * 0.3 + cv_current_x[1] * 0.7, cv_current_y],
                                   [cv_current_x[0] * 0.7 + cv_current_x[1] * 0.3, cv_current_y],
                                   cv_current_end])
        self.current_cv, _ = smooth_ployline(current_cv_raw)

    def idm_planner(self, given_leading_car=None):
        self.get_leading_vehicle_info(given_leading_car)
        acc = idm_model(self.idm_parameter,
                        self.vx,
                        self.delta_speed_from_front,
                        self.distance_from_front)
        self.vx += acc * dt
        self.x += self.vx * dt

    def draw_box(self, ax):
        draw_rectangle(self.x, self.y, 0, ax, para_alpha=1, para_color='#0E76CF')
        ax.text(self.x, self.y + 1, str(self.global_id), size=10, color='black')
        ax.text(self.x + 6, self.y + 1, str("%.2f" % self.vx), size=8, color='green')
        # plt.savefig('test')


class AutonomousVehicle(Vehicle):
    def __init__(self, source_lane, road_net, controller, ipv):
        super().__init__(source_lane, road_net)
        self.is_planning_back = False
        self.is_av = True
        self.surrounding_vehicles = {}
        self.target_cv = None
        self.controller = controller
        self.ipv = ipv * math.pi / 4
        self.trajectory_solution = np.array([])

    def update_av_motion(self):
        """
        motion planner settings
        """
        if self.controller == 'IDM':
            self.idm_planner()
        if self.controller == 'NGMP':  # Nash game motion planner
            self.ngmp_planner()
        if self.controller == 'Lattice':
            self.lattice_planner()


    def lattice_planner(self):
        agent_av = Agent(
            [self.x, self.y],  # position
            [self.vx, self.vy],  # velocity
            self.heading,  # heading
            self.target_cv  # central vertices of the target lane
        )
        agent_av.ipv = self.ipv
        bv_rightback = self.surrounding_vehicles['rightback']
        bv_rightfront = self.surrounding_vehicles['rightfront']  # 如果直接忽略前车可能会变道时追尾 所以与右车道前车距离近时 要视为障碍物
        bv_front = self.surrounding_vehicles['front']  # 如果同车道前车比自己慢 且在反应距离内 则自己的规划速度不能大于前车当前速度 不用视为障碍物 但规划时需要减速

        self.v = math.sqrt(self.vx ** 2 + self.vy ** 2)
        traj_point = TrajPoint(
            [self.x, self.y, self.v, 0,
             self.heading, 0])  # [x y v a theta kappa] [x y v theta]

        # print(self.x - bv_rightback.x, bv_rightfront.x-self.x, bv_front.x - self.x, self.x)
        # get obs
        obstacles = []
        planning_horizon = 15  # 后车规划
        already_planned = 0
        for obs in range(planning_horizon):
            obs_point = [bv_rightback.x + already_planned * bv_rightback.vx * 0.15,
                         bv_rightback.y + already_planned * bv_rightback.vy * 0.15,
                         math.sqrt(bv_rightback.vx ** 2 + bv_rightback.vy ** 2),
                         bv_rightback.heading]  # 上时刻在obs时刻的规划轨迹 数据为 x y v theta
            obstacles.append(Obstacle([obs_point[0], obs_point[1], 0, VEH_L, VEH_W, obs_point[3]]))
            already_planned += 1

        already_planned = 0
        if bv_rightfront.x - self.x < VEH_L:  # 右前车还很近 应该被视为障碍物
            print('right front 有车， 且被视为障碍物')
            for obs in range(15):
                obs_point = [bv_rightfront.x + already_planned * bv_rightfront.vx * 0.15 - 2 * VEH_L,
                             bv_rightfront.y + already_planned * bv_rightfront.vy * 0.15,
                             math.sqrt(bv_rightfront.vx ** 2 + bv_rightfront.vy ** 2),
                             bv_rightfront.heading]  # 当前位置即可 不用带提前量 * Dt 不差这点
                obstacles.append(Obstacle([obs_point[0], obs_point[1], 0, VEH_L, VEH_W, obs_point[3]]))
                already_planned += 1

        # backup 4 bv_rightfront 目前无用
        # already_planned = 0
        # if bv_front.x - self.x < VEH_L:  # 当本车侵入右车道时 右前车变为前车还很近 应该被视为障碍物
        #     for obs in range(planning_horizon):
        #         obs_point = [bv_front.x + already_planned * bv_front.vx * 0.1,
        #                     bv_front.y + already_planned * bv_front.vy * 0.1,
        #                     math.sqrt(bv_front.vx ** 2 + bv_front.vy ** 2),
        #                     bv_front.heading]  # 当前位置即可 不用带提前量 * Dt 不差这点
        #         obstacles.append(Obstacle([obs_point[0], obs_point[1], 0, VEH_L, VEH_W, obs_point[3]]))
        #         already_planned += 1

        if -0.05 < self.y < 0.05:  # 已经在最后一个车道时
            path_points = CalcRefLine(straight_line([self.x, self.y]).T)  # 这一步改变ref_line 所以不对self.ref_line调整
            for obstacle in obstacles:
                obstacle.MatchPath(path_points)  ### 同样match障碍物与参考轨迹
            traj_point.MatchPath(path_points)
            samp_basis = SampleBasis(traj_point, theta_thr, ttcs)
            local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis, self.ipv)
            if bv_front != None:
                if bv_front.x - self.x < sight_range and bv_front.vx < self.vx:  # 规划速度不超过前车速度
                    print('存在前车， 速度不能超过前车')
                    traj_points_opt, r = local_planner.LocalPlanning(traj_point, path_points, target_v=bv_front.vx)
                else:
                    traj_points_opt, r = local_planner.LocalPlanning(traj_point, path_points, target_v=v_tgt)
            else:
                traj_points_opt, r = local_planner.LocalPlanning(traj_point, path_points, target_v=v_tgt)

        else:
            save_r = []
            save_traj_opt = []
            steep_list = [0.05, 0.1, 0.2]
            for steep in steep_list:
                # get ref_line
                ref_start_lane = math.ceil(self.y / 4)  # y [12, 8) lane = 3; y [8, 4) lane = 2
                ref_line = sigmoid_function([self.x, ref_start_lane * 4], steepness=steep).T
                gap = ref_line[0][np.argmin(
                    abs(self.y - ref_line[1]))] - self.x  # 解决闪现问题 由于中途更换steep导致目前车辆位置可能不在ref_line上 所以直接match会出现闪现
                # print(self.y, ref_start_lane, np.argmin(abs(self.y - ref_line[1])))
                ref_line[0] = ref_line[0] - gap  # 解决方案是将现在的ref_line平移到当前位置
                path_points = CalcRefLine(ref_line)

                # match data
                for obstacle in obstacles:
                    obstacle.MatchPath(path_points)  ### 同样match障碍物与参考轨迹
                    # plt.gca().add_patch(plt.Rectangle((obstacle.corner[0], obstacle.corner[1]), obstacle.length,
                    #                                    obstacle.width, color='r', angle=obstacle.heading*180/M_PI))
                    # plt.axis('scaled')

                traj_point.MatchPath(path_points)  # matching once is enough 将traj_point(单点)与path_points(序列)中最近的点匹配

                samp_basis = SampleBasis(traj_point, theta_thr, ttcs)
                local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis, self.ipv)
                if bv_front != None:
                    if bv_front.x - self.x < sight_range and bv_front.vx < self.vx:
                        print('存在前车， 速度不能超过前车')
                        traj_points_opt, r = local_planner.LocalPlanning(traj_point, path_points, target_v=bv_front.vx)
                    else:
                        traj_points_opt, r = local_planner.LocalPlanning(traj_point, path_points, target_v=v_tgt)
                else:
                    traj_points_opt, r = local_planner.LocalPlanning(traj_point, path_points, target_v=v_tgt)

                save_traj_opt.append(traj_points_opt)
                save_r.append(r)

            best_traj = np.argmax(save_r)
            # print('当前规划的steepness为', steep_list[best_traj], '(from 0.1-0.3)', save_r)
            traj_points_opt = save_traj_opt[best_traj]

            if len(traj_points_opt) <= 10:
                # print("无解, 考虑直行")
                path_points = CalcRefLine(straight_line([self.x, self.y]).T)  # 这一步改变ref_line 所以不对self.ref_line调整
                for obstacle in obstacles:
                    obstacle.MatchPath(path_points)  ### 同样match障碍物与参考轨迹
                traj_point.MatchPath(path_points)
                samp_basis = SampleBasis(traj_point, theta_thr, ttcs)
                local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis, self.ipv)
                traj_points_opt, r = local_planner.LocalPlanning(traj_point, path_points, target_v=0)

        traj_points = []
        for tp_opt in traj_points_opt:
            traj_points.append(
                [tp_opt.x, tp_opt.y, tp_opt.v * math.sin(tp_opt.theta), tp_opt.v * math.cos(tp_opt.theta),
                 tp_opt.theta])
        self.trajectory_solution = np.array(traj_points)
        self.x = traj_points_opt[1].x
        self.y = traj_points_opt[1].y
        self.heading = traj_points_opt[1].theta
        self.vx = traj_points_opt[1].v * math.sin(self.heading)
        self.vy = traj_points_opt[1].v * math.cos(self.heading)

    def ngmp_planner(self):
        try:
            self.planning_to_target_cv()

            if self.check_collision():
                print('planning to target failed, keep in current lane')
                self.planning_to_current_cv()
                self.is_planning_back = True

                if self.check_collision(target={'front'}):
                    print('keep in current lane failed, start car-following')
                    self.idm_planner()
                else:
                    self.get_new_state()

            else:
                self.is_planning_back = False
                self.get_new_state()
        except:
            print('NGMP planning failed')
            self.idm_planner()

    def planning_to_target_cv(self):

        agent_av = Agent(
            [self.x, self.y],  # position
            [self.vx, self.vy],  # velocity
            self.heading,  # heading
            self.target_cv  # central vertices of the target lane
        )
        agent_av.ipv = self.ipv

        bv = self.surrounding_vehicles['rightback']
        bv.get_central_vertices()
        agent_bv = Agent(
            [bv.x, bv.y],  # position
            [bv.vx, bv.vy],  # velocity
            bv.heading,  # heading
            bv.current_cv
        )
        agent_av.estimated_inter_agent = [copy.deepcopy(agent_bv)]
        agent_av.lp_ibr_interact(iter_limit=50, interactive=True)
        self.trajectory_solution = agent_av.trj_solution[:, 0:5]

    def planning_to_current_cv(self):
        agent_av = Agent(
            [self.x, self.y],  # position
            [self.vx, self.vy],  # velocity
            self.heading,  # heading
            self.current_cv  # central vertices of the target lane
        )
        agent_av.ipv = self.ipv

        agent_bv = Agent(
            [0, self.y],  # position
            [0, 0],  # velocity
            0,  # heading
            self.current_cv
        )

        agent_av.estimated_inter_agent = [copy.deepcopy(agent_bv)]
        agent_av.lp_ibr_interact(iter_limit=50, interactive=True)
        self.trajectory_solution = agent_av.trj_solution[:, 0:5]

    def get_new_state(self):
        self.x = self.trajectory_solution[1, 0]
        self.y = self.trajectory_solution[1, 1]
        self.vx = self.trajectory_solution[1, 2]
        self.vy = self.trajectory_solution[1, 3]
        self.heading = self.trajectory_solution[1, 4]

    def check_collision(self, target=None):
        av_trj = np.array(self.trajectory_solution[1:, 0:2])
        if target:
            for veh_target in target:
                veh = self.surrounding_vehicles[veh_target]
                if veh:
                    future_x = veh.x + np.cumsum(
                        veh.vx * dt * np.ones([np.size(self.trajectory_solution, 0) - 1, 1])) - veh.vx * dt
                    future_trj = np.array([[x, veh.y] for x in list(future_x)])
                    distance = np.linalg.norm(av_trj - future_trj, axis=1)
                    if min(distance) < 5:
                        # print('collision to vehicle: ', veh.global_id)
                        return True
        else:
            for veh in self.surrounding_vehicles.values():
                if veh:
                    future_x = veh.x + np.cumsum(
                        veh.vx * dt * np.ones([np.size(self.trajectory_solution, 0) - 1, 1])) - veh.vx * dt
                    future_trj = np.array([[x, veh.y] for x in list(future_x)])
                    distance = np.linalg.norm(av_trj - future_trj, axis=1)
                    if min(distance) < 5:
                        # print('collision to vehicle: ', veh.global_id)
                        return True
        return False

    def get_surrounding_vehicle(self, rn):
        self.surrounding_vehicles['front'] = None
        self.surrounding_vehicles['back'] = None

        x_position_list = []
        for veh in rn.lane_list[self.lane_id].vehicle_list:
            x_position_list.append(veh.x - self.x)
        x_position_arr = np.array(x_position_list)

        if max(x_position_arr) > 0:
            min_positive_index = np.where(x_position_arr == min(x_position_arr[x_position_arr > 0]))
            self.surrounding_vehicles['front'] = rn.lane_list[self.lane_id].vehicle_list[
                min_positive_index[0][0]]
        if min(x_position_arr) < 0:
            max_negative_index = np.where(x_position_arr == max(x_position_arr[x_position_arr < 0]))
            self.surrounding_vehicles['back'] = rn.lane_list[self.lane_id].vehicle_list[
                max_negative_index[0][0]]

        if not self.lane_id == 0:  # not at the down first lane, right lane exists
            self.surrounding_vehicles['rightfront'] = None
            self.surrounding_vehicles['rightback'] = None

            x_position_list = []
            for veh in rn.lane_list[self.lane_id - 1].vehicle_list:
                x_position_list.append(veh.x - self.x)
            x_position_arr = np.array(x_position_list)

            if max(x_position_arr) > 0:
                min_positive_index = np.where(x_position_arr == min(x_position_arr[x_position_arr > 0]))
                self.surrounding_vehicles['rightfront'] = rn.lane_list[self.lane_id - 1].vehicle_list[
                    min_positive_index[0][0]]
            if min(x_position_arr) < 0:
                max_negative_index = np.where(x_position_arr == max(x_position_arr[x_position_arr < 0]))
                self.surrounding_vehicles['rightback'] = rn.lane_list[self.lane_id - 1].vehicle_list[
                    max_negative_index[0][0]]

    def get_av_central_vertices(self, rn):

        # get current cv
        cv_current_x = rn.lane_list[self.lane_id].x_range
        cv_current_y = np.mean(rn.lane_list[self.lane_id].y_range)
        cv_current_start = [cv_current_x[0], cv_current_y]
        cv_current_end = [cv_current_x[1], cv_current_y]
        current_cv_raw = np.array([cv_current_start,
                                   [cv_current_x[0] * 0.3 + cv_current_x[1] * 0.7, cv_current_y],
                                   [cv_current_x[0] * 0.7 + cv_current_x[1] * 0.3, cv_current_y],
                                   cv_current_end])
        self.current_cv, _ = smooth_ployline(current_cv_raw)

        # get target cv
        if self.lane_id == 0:
            cv_target_y = cv_current_y
        else:
            cv_target_y = self.y - 1
        cv_target_x = rn.lane_list[self.lane_id - 1].x_range
        cv_target_start = [cv_target_x[0], cv_target_y]
        cv_target_end = [cv_target_x[1], cv_target_y]
        target_cv_raw = np.array([cv_target_start,
                                  [cv_target_x[0] * 0.3 + cv_target_x[1] * 0.7, cv_target_y],
                                  [cv_target_x[0] * 0.7 + cv_target_x[1] * 0.3, cv_target_y],
                                  cv_target_end])
        self.target_cv, _ = smooth_ployline(target_cv_raw)

    def check_lane(self, rn):
        new_lane_id = max(0, int(np.floor((self.y + 2) / 4)))
        if new_lane_id == self.lane_id:
            return False
        else:
            if min(abs(np.array(rn.lane_list[new_lane_id].y_range) - self.y)) < 0.5:
                return False
            else:
                self.last_lane_id = self.lane_id
                self.lane_id = new_lane_id
                self.source_lane = rn.lane_list[new_lane_id]

                # update vehicle list of affected lanes
                self.source_lane.vehicle_list.append(self)
                rn.lane_list[self.last_lane_id].vehicle_list.remove(self)
                return True

    def draw_box(self, ax):
        draw_rectangle(self.x, self.y, self.heading, ax, para_alpha=1, para_color='red')
        ax.text(self.x, self.y + 1, str(self.global_id), size=10)
        ax.text(self.x + 6, self.y + 1, str("%.2f" % self.vx), size=8, color='green')
        if len(self.trajectory_solution):
            ax.plot(self.trajectory_solution[1:, 0], self.trajectory_solution[1:, 1], 'red')


def idm_model(para, vel_self, vel_rel, gap):
    akgs = vel_self * para[1] + vel_self * vel_rel / 2 / np.sqrt(para[2] * para[3])
    if akgs < 0:
        sss = para[0]
    else:
        sss = para[0] + akgs

    acc = para[2] * (1 - np.power((vel_self / para[4]), 5) - np.power((sss / gap), 2))
    if acc > para[2]:
        acc = para[2]
    if acc < -para[2]:
        acc = -para[2]

    return acc


def draw_rectangle(x, y, deg, ax, para_alpha=0.5, para_color='blue'):
    car_len = 2
    car_wid = 4

    r2 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color=para_color, alpha=para_alpha)

    t2 = mt.Affine2D().rotate_deg_around(x, y, deg) + ax.transData
    r2.set_transform(t2)

    ax.add_patch(r2)
