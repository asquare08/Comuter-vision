import argparse
import cv2
import imutils
import numpy as np
from pipeline import (VehicleDetectorOpticalFlow, VehicleDetectorBackgroundSubtraction,
                      LicencePlateDetector)


class LicencePlateRecognitionApp:

    def __init__(self):

        self.capture = None
        self.vehicle_detector = None
        self.plate_detector = None
        self.frame = None
        self.car = None

        self.capture = cv2.VideoCapture(args['input-file'].name)
        frame, frame_gray = self.next_frame()

        self.vehicle_detector = VehicleDetectorBackgroundSubtraction(frame_gray)
        self.plate_detector = LicencePlateDetector()

        cv2.namedWindow('frame')
        cv2.moveWindow('frame', 0, 50)
        cv2.namedWindow('plate')
        cv2.moveWindow('plate', 1024, 50)
    def next_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            LicencePlateRecognitionApp().run()
            # exit("Video is empty")
        frame = frame[15:, :]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, frame_gray

    def run(self):
        num = 0
        num1 = 0
        # temp = (400, 800)
        # plate_tresh = np.ones(temp)*255
        # mask = plate_tresh
        while self.capture.isOpened():
            frame, frame_gray = self.next_frame()

            vehicles = self.vehicle_detector.next_frame(frame_gray)
            # cv2.imshow('vehicles', vehicles)
            for vehicle in vehicles:
                car_grey = frame_gray[vehicle[0][1]:vehicle[1][1], vehicle[0][0]:vehicle[1][0]]
                plate_rect = self.plate_detector.detect(car_grey)

                cv2.rectangle(frame, vehicle[0], vehicle[1], (0, 255, 0), 3)
                if plate_rect:
                    x, y, w, h = plate_rect
                    plate_grey = car_grey[y:y+h, x:x+w]
                    plate_tresh = cv2.adaptiveThreshold(plate_grey, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY_INV, 13, 4)
                    # cv2.imwrite('output\plate\pic'+str(num)+'.jpg', plate_tresh)
                    num += 1
                    cv2.imshow('plate', plate_tresh)
                    x += vehicle[0][0]
                    y += vehicle[0][1]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

            small_frame = imutils.resize(frame, width=1024)
            # cv2.imwrite('output\cars\pic'+str(num1)+'.jpg', small_frame)
            # cv2.imwrite('output\cars\pic_plate'+str(num1)+'.jpg', plate_tresh)
            cv2.imshow("frame", small_frame)
            # [row, col] = plate_tresh.shape
            # mask[0:row, 0:col] = plate_tresh
            # cv2.imshow('final', mask)
            # cv2.imwrite('output\cars\pic_plate_new' + str(num1) + '.jpg', mask)
            # print small_frame.shape
            # vis = np.concatenate((small_frame, mask), axis=1)
            # cv2.imshow('final', vis)
            # mask = np.ones(temp) * 255
            num1 += 1
            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                exit()
            if k == ord('r'):
                LicencePlateRecognitionApp().run()
            if k == ord('p'):
                while True:
                    x = cv2.waitKey(1) & 0xff
                    if x == ord('c'):
                        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic License Plate Recognition')
    parser.add_argument('input-file', type=argparse.FileType('r'), help='path to video input file')
    args = vars(parser.parse_args())

    LicencePlateRecognitionApp().run()



