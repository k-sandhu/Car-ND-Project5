import numpy as np

class Cars():

    def __init__(self):
        self.data = [] # List of current_data dictionaries for previous frames

        self.current_data = {
            "frame_num": 0,
            "num_cars": 0,
            "num_false_positives": 0,
            "locations": [],  # location in the form [left_x_cord, top_y_cord, right_x_coord, bottom_y_coord]
            "false_positive": [],
            "b_boxes":[],
            "heat_map": np.zeros(shape=(720,1280,3))
        }

    def false_positive(self, location):
        """
        Detects false positives by comparing the locations of the detected cars to the
        previous frame of the video. If the car was not present in the previous frame, it
        is tagged as a false positive.
        :param location:
        :return:
        """
        is_false_positive = False
        return is_false_positive

    def add_locations(self, frame_num, locations):
        """
        Adds location to the car to locations.
        :param frame_num:
        :param locations: [left_x_cord, top_y_cord, right_x_coord, bottom_y_coord]
        :return: None
        """
        self.current_data = {
            "frame_num": 0,
            "num_cars": 0,
            "num_false_positives": 0,
            "locations": [],  # location in the form [left_x_cord, top_y_cord, right_x_coord, bottom_y_coord]
            "false_positive": [],
            "b_boxes":[],
            "heat_map": np.zeros(shape=(720,1280,3))
        }

        temp_loc = []
        for location in locations:
            if not self.false_positive(location):
                temp_loc.append(location)
            else: self.current_data["false_positive"].append(location)

        self.current_data["locations"] = temp_loc
        self.current_data["num_cars"] = len(self.current_data["locations"])
        self.current_data["num_false_positives"] = len(self.current_data["false_positive"])

        self.make_heat_map()

        self.data.append(self.current_data)
        #self.current_data["b_boxes"] = self.make_b_boxes()
        self.print_attr()

        return self.current_data

    def make_b_boxes(self):
        b_boxes = []
        return b_boxes

    def make_heat_map(self):
        """
        Draws a heat map using current vehicle locations and 3 channel np.zeros array.
        :return: None
        """
        for box in self.current_data["locations"]:
            self.current_data["heat_map"][box[1]:box[3],box[0]:box[2],0] += 1

        self.current_data["heat_map"] = self.current_data["heat_map"]/np.max(self.current_data["heat_map"])
        print(np.max(self.current_data["heat_map"]), np.mean(self.current_data["heat_map"]),np.std(self.current_data["heat_map"]))


    def get_heat_map(self):
        """
        Draws a heat map using current vehicle locations and 3 channel np.zeros array.
        :return: None
        """
        return self.current_data["heat_map"]

    def print_attr(self):
        print(self.current_data["frame_num"])
        print(len(self.current_data["locations"]))

