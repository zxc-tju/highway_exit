import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as mt
from agent import Agent
from tools.utility import smooth_ployline

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

    def idm_update(self, given_leading_car=None):
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

        if self.controller == 'IDM':
            self.idm_update()
        if self.controller == 'NGMP':  # Nash game motion planner
            self.ngmp_update()

    def ngmp_update(self):
        self.planning_to_target_cv()

        if self.check_collision():
            print('planning to target failed, keep in current lane')
            self.planning_to_current_cv()
            self.is_planning_back = True

            if self.check_collision(target={'front'}):
                print('keep in current lane failed, start car-following')
                self.idm_update()
            else:
                self.get_new_state()

        else:
            self.is_planning_back = False
            self.get_new_state()

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
                    future_x = veh.x + np.cumsum(veh.vx * dt * np.ones([np.size(self.trajectory_solution, 0) - 1, 1])) - veh.vx * dt
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
        for veh in rn.lane_list[self.lane_id - 1].vehicle_list:
            x_position_list.append(veh.x - self.x)
        x_position_arr = np.array(x_position_list)

        if max(x_position_arr) > 0:
            min_positive_index = np.where(x_position_arr == min(x_position_arr[x_position_arr > 0]))
            self.surrounding_vehicles['front'] = rn.lane_list[self.lane_id - 1].vehicle_list[
                min_positive_index[0][0]]
        if min(x_position_arr) < 0:
            max_negative_index = np.where(x_position_arr == max(x_position_arr[x_position_arr < 0]))
            self.surrounding_vehicles['back'] = rn.lane_list[self.lane_id - 1].vehicle_list[
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
                self.surrounding_vehicles['rightfront'] = rn.lane_list[self.lane_id - 1].vehicle_list[min_positive_index[0][0]]
            if min(x_position_arr) < 0:
                max_negative_index = np.where(x_position_arr == max(x_position_arr[x_position_arr < 0]))
                self.surrounding_vehicles['rightback'] = rn.lane_list[self.lane_id - 1].vehicle_list[max_negative_index[0][0]]

    def get_av_central_vertices(self, rn):
        cv_current_x = rn.lane_list[self.lane_id].x_range
        cv_current_y = np.mean(rn.lane_list[self.lane_id].y_range)
        cv_current_start = [cv_current_x[0], cv_current_y]
        cv_current_end = [cv_current_x[1], cv_current_y]
        current_cv_raw = np.array([cv_current_start,
                                   [cv_current_x[0] * 0.3 + cv_current_x[1] * 0.7, cv_current_y],
                                   [cv_current_x[0] * 0.7 + cv_current_x[1] * 0.3, cv_current_y],
                                   cv_current_end])
        self.current_cv, _ = smooth_ployline(current_cv_raw)

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
