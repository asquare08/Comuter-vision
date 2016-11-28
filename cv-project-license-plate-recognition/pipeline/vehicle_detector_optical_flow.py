import numpy as np
import cv2
import itertools


class VehicleDetectorOpticalFlow:

    def __init__(self, init_frame_gray):
        self.last_frame_gray = init_frame_gray.copy()

        self.feature_params = dict(maxCorners=300, qualityLevel=0.6, minDistance=5, blockSize=7)
        self.p0 = cv2.goodFeaturesToTrack(init_frame_gray, mask=None, **self.feature_params)

        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.color = np.random.randint(0, 255, (100, 3))

    def next_frame(self, frame_gray):
        vehicles = []
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame_gray, frame_gray, self.p0, None, **self.lk_params)

        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        dist = np.array([np.linalg.norm(d) for d in good_old - good_new])

        moving_points = good_old[dist > 1]

        if len(moving_points):
            x = moving_points[:, 0]
            y = moving_points[:, 1]
            vehicles.append([(min(x), min(y)), (max(x), max(y))])

        self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        print self.p0
        self.last_frame_gray = frame_gray.copy()

        return vehicles



