from vehicle import Vehicle, AutonomousVehicle


class RoadNetwork:
    def __init__(self):
        self.lane_list = []  # list of Lane class
        self.vehicle_list = []  # list of Vehicle class
        self.traffic_quantity = 30  # x vehicles per lane per kilometer
        self.vehicle_number = 0
        self.lane_number = 4

    def update_rank_in_lane(self, av: AutonomousVehicle):
        old_lane_id = av.last_lane_id
        new_lane_id = av.lane_id
        self.lane_list[old_lane_id].vehicle_number -= 1
        self.lane_list[new_lane_id].vehicle_number += 1

        old_rank = av.rank_in_lane
        if av.surrounding_vehicles['rightback']:
            new_rank = av.surrounding_vehicles['rightback'].rank_in_lane
        else:  # av is the newest car in the new lane
            new_rank = self.lane_list[new_lane_id].vehicle_number
        av.rank_in_lane = new_rank

        for veh in self.lane_list[old_lane_id].vehicle_list:
            if veh.rank_in_lane > old_rank:  # and not veh.is_av
                veh.rank_in_lane -= 1

        av.back_ttc = None
        av.front_ttc = None
        for veh in self.lane_list[new_lane_id].vehicle_list:
            if veh.rank_in_lane >= new_rank and not veh.is_av:
                veh.rank_in_lane += 1
                if veh.rank_in_lane == new_rank + 1:  # find following car
                    av.back_ttc = (av.x - veh.x) / (veh.vx - av.vx)
            if veh.rank_in_lane == new_rank - 1:  # find leading car
                av.front_ttc = (veh.x - av.x) / (av.vx - veh.vx)


class Lane:
    def __init__(self, lane_id=0, traffic_quantity=30, lane_len=500):
        self.id = lane_id
        self.length = lane_len
        self.traffic_quantity = traffic_quantity
        self.x_range = [0, 3000]
        self.y_range = [lane_id * 4 - 2, lane_id * 4 + 2]

        self.vehicle_list = []
        self.vehicle_number = 0

    def add_vehicle(self, road_net):
        self.vehicle_list.append(
                    Vehicle(self, road_net)
                )

