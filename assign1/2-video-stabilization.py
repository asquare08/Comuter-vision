import numpy as np
import cv2
import lucas_kanade_impl as lk

# colors
red_bgr = (0, 0, 255)
color = np.random.randint(0, 255, (100, 3))

# params for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def main():
    ret, old_frame = cap.read()
    if not ret:
        exit("Video is empty")

    first_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)
    if p0 is None:
        exit("No corners found")

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pi, st, _ = cv2.calcOpticalFlowPyrLK(first_gray, current_gray, p0, None, **lk_params)
        # pi, st = lk.affinelk(first_gray, current_gray, p0)

        H, mask = cv2.findHomography(pi, p0, method=cv2.RANSAC)
        result = cv2.warpPerspective(frame, H, (640, 360))

        cv2.imshow('shaky', frame)

        cv2.imshow('stabilized', result)
        cv2.imwrite('output/img_'+str(i).zfill(4)+'.jpg', result)
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break

        i += 1


if __name__ == '__main__':
    cap = cv2.VideoCapture('data/shaky.mov')

    main()

    cap.release()
    cv2.destroyAllWindows()