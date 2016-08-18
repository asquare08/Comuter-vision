import numpy as np
import cv2
import lucas_kanade_impl as lk
import os

# colors
red_bgr = (0, 0, 255)
color = np.random.randint(0, 255, (100, 3))

# params for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def track(old_gray, p0):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # p1, st = lk.lucasKanadeICAffine(old_gray, frame_gray, p0)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
            cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('r'):
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
            main()
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


def main():
    ret, frame = cap.read()
    if not ret:
        exit("Video is empty")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    if p0 is None:
        exit("No corners found")

    for i in p0:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 3, 255, -1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    track(gray, p0)


if __name__ == '__main__':
    cap = cv2.VideoCapture('data/traffic-scene.mp4')

    main()

    cap.release()
    cv2.destroyAllWindows()
