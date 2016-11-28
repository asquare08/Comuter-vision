import numpy as np
import cv2
import itertools


class VehicleDetectorBackgroundSubtraction:

    def __init__(self, background_frame):
        self.background_frame = background_frame.copy()

        self.threshold = 70

    def next_frame(self, frame_gray):
        vehicles = []

        diff = cv2.absdiff(self.background_frame, frame_gray)
        diff[diff <= self.threshold] = 0

        objects = np.nonzero(diff)
        if objects[0].size:
            x = objects[0]
            y = objects[1]
            vehicles.append([(min(y), min(x)), (max(y), max(x))])

        return vehicles
