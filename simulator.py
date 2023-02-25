from roadnet import Lane, RoadNetwork
from vehicle import Vehicle, AutonomousVehicle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
import xlsxwriter
from tqdm import tqdm
from time import gmtime, strftime

dt = 0.1
AV_controller = 'NGMP'  # 'NGMP'  'IDM'  'Lattice'
AV_IPV = 0
WARMUP_TIME = 5

SIMULATION_FRAMES = 500
LANE_LEN = 1000
save_data_needed = True


class Simulator:
    def __init__(self, case_id):
        self.case_id = case_id
        self.fig, self.ax = plt.subplots(figsize=(18, 5))
        self.time = 0
        self.road_net = RoadNetwork()
        self.av_exist = False
        self.av = None
        self.is_task_complete = False
        self.record_data = {
            'case': self.case_id,
            'success?': 0,
            'finish time': 0,
            'finish x': None,
            'f_ttc_1': None,
            'b_ttc_1': None,
            'f_ttc_2': None,
            'b_ttc_2': None,
            'f_ttc_3': None,
            'b_ttc_3': None,
        }

    def initialize(self):
        self.initialize_lane_list()
        self.initialize_vehicle_list()
        self.draw_road_net()
        ani = FuncAnimation(self.fig, self.update, interval=10,
                            frames=SIMULATION_FRAMES, blit=False, repeat=False, save_count=SIMULATION_FRAMES)

        " ---- option 1: show animation ---- "
        # plt.show()

        " ---- option 2: save as gif ---- "
        # ani.save('test.gif', fps=10, dpi=300)

        # " ---- option 3: save as mp4 video ---- "
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, codec="h264", bitrate=-1, metadata=dict(dpi=600, artist='Me'))
        video_dir = '../outputs/highway_exit/video/' + str(LANE_LEN) + 'm/' + \
                    start_time + '-' + AV_controller + '-ipv-' + str(AV_IPV) + '/'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        ani.save(video_dir + str(self.case_id) + '.mp4', writer=writer)
        plt.close()

    def update(self, frame):

        self.time += dt
        self.update_fig()
        self.update_vehicle_motion()  # update both AV and background vehicles
        if self.av and not self.is_task_complete:
            # check whether task is finished. if yes, save position and time step.
            self.check_task()

        if frame == SIMULATION_FRAMES - 1 and save_data_needed:
            # run out of time, save task info.
            self.save_data()

    def save_data(self):
        file_name = '../outputs/highway_exit/data/' + str(LANE_LEN) + 'm/' + \
                    start_time + '-' + AV_controller + '-ipv-' + str(AV_IPV) + '.xlsx'
        if not os.path.exists(file_name):
            workbook = xlsxwriter.Workbook(file_name)
            workbook.add_worksheet()
            workbook.close()

        df = pd.DataFrame([[v for v in self.record_data.values()], ], columns=[d for d in self.record_data.keys()])
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists="overlay", engine="openpyxl") as writer:
            # writer.book = book
            if self.case_id == 0:
                df.to_excel(writer, header=True, index=False, startcol=0, startrow=0)
            else:
                df.to_excel(writer, header=False, index=False, startcol=0, startrow=self.case_id + 1)
        # print('finished')

    def check_task(self):
        # check if task completed
        if self.av.lane_id == 0:
            self.is_task_complete = True
            self.record_data['finish time'] = self.time
            self.record_data['finish x'] = self.av.x
            if self.av.x < self.av.source_lane.length:
                self.record_data['success?'] = 1

    def update_vehicle_motion(self):
        # update background vehicles
        self.update_background_vehicle()
        self.add_new_vehicle()

        if self.time > WARMUP_TIME:

            if not self.av_exist:
                self.add_av()
            else:
                self.update_av()

    def add_av(self):
        lane = self.road_net.lane_list[-1]
        if lane.vehicle_list[-1].x > 30:
            self.av = AutonomousVehicle(lane, self.road_net, AV_controller, AV_IPV)
            lane.vehicle_list.append(self.av)
            self.av_exist = True
            self.av.get_av_central_vertices(self.road_net)
            self.av.get_surrounding_vehicle(self.road_net)

    def update_av(self):
        if self.time > WARMUP_TIME + 5:  # start control after a further while
            self.av.get_surrounding_vehicle(self.road_net)
            if self.av.check_lane(self.road_net):  # check if av has changed lane
                self.road_net.update_rank_in_lane(self.av)
                self.record_ttc_after_lc()

            # if not self.av.is_planning_back:  # if AV is planning back to current lane, do not change target lane
            self.av.get_av_central_vertices(self.road_net)

            self.av.update_av_motion()

        else:
            self.av.idm_planner()

        self.av.draw_box(self.ax)

    def update_background_vehicle(self):
        for lane in self.road_net.lane_list:
            for veh_id, veh in enumerate(lane.vehicle_list):
                if self.av:
                    if self.av.surrounding_vehicles['rightback']:
                        if veh.global_id == self.av.surrounding_vehicles['rightback'].global_id:
                            if self.av.y - veh.y < veh.react_threshold:
                                veh.idm_planner(given_leading_car=self.av)
                                veh.draw_box(self.ax)
                                continue
                if not veh.is_av:
                    veh.idm_planner()
                    veh.draw_box(self.ax)
                    if veh.x > lane.length:
                        lane.vehicle_list.pop(0)  # delete vehicles run out of the lane

    def add_new_vehicle(self):
        for lane in self.road_net.lane_list:
            if len(lane.vehicle_list) < lane.traffic_quantity:
                if lane.vehicle_list[-1].x > 30 + np.random.uniform(-1, 1) and np.random.uniform(0, 1) > 0.9:
                    lane.add_vehicle(self.road_net)

    def draw_road_net(self):
        for lane in self.road_net.lane_list:
            self.ax.plot(lane.x_range,
                         [np.mean(lane.y_range), np.mean(lane.y_range)],
                         color='gray')

    def record_ttc_after_lc(self):
        if self.av.back_ttc:
            self.record_data['b_ttc_' + str(int(3 - self.av.lane_id))] = self.av.back_ttc
        if self.av.front_ttc:
            self.record_data['f_ttc_' + str(int(3 - self.av.lane_id))] = self.av.front_ttc

    def initialize_lane_list(self):
        for i in range(self.road_net.lane_number):
            # create lanes
            self.road_net.lane_list.append(
                Lane(lane_id=i, traffic_quantity=self.road_net.traffic_quantity, lane_len=LANE_LEN))

    def initialize_vehicle_list(self):
        for lane in self.road_net.lane_list:  # create vehicles for each lane
            lane.vehicle_list.append(
                Vehicle(lane, self.road_net)
            )

    def update_fig(self):
        self.ax.cla()
        self.ax.axis('scaled')
        self.ax.set_xlim(0, self.road_net.lane_list[0].length)
        self.ax.set_ylim(-4, self.road_net.lane_number * 4)
        # self.ax.text(20, 20, 'time=' + str(int(self.time)), size=20)

        if self.av:
            self.ax.text(self.av.x, 20, 'time=' + str(int(self.time)), size=20)
            self.ax.set_xlim(max(0, self.av.x - 100), self.av.x + 100)
            self.ax.plot(self.av.target_cv[:, 0], self.av.target_cv[:, 1], color='blue', linewidth=2)
            self.ax.plot(self.av.current_cv[:, 0], self.av.current_cv[:, 1], color='green', linewidth=2)
        self.draw_road_net()

        # Set the borders to a given color
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
        plt.yticks([])


if __name__ == '__main__':
    # " single run "
    # simu = Simulator(0)
    # simu.initialize()

    " multi runs "
    start_time = strftime("%Y-%m-%d-%H", gmtime())
    print('start:' + AV_controller + '-ipv-' + str(AV_IPV) + '-' + start_time)
    proc_bar = tqdm(range(0, 100))
    for n in proc_bar:
        try:
            simu = Simulator(n)
            simu.initialize()
        except:
            print('case', str(n), 'failed')
            continue
