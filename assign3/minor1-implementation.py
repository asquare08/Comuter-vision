import numpy as np
import cv2
cap = cv2.VideoCapture('video.avi')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
# Select good points
good_new = p1[st == 1]
good_old = p0[st == 1]
velocity = (good_new - good_old)*10
u1 = velocity[:, 0]
v1 = velocity[:, 1]
while 1:
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    u1 = np.array([u1]).T[st == 1]
    v1 = np.array([v1]).T[st == 1]
    velocity = (good_new - good_old) * 10
    u2 = velocity[:, 0]
    v2 = velocity[:, 1]
    x1 = good_old[:, 0]
    y1 = good_old[:, 1]
    x2 = good_new[:, 0]
    y2 = good_new[:, 1]
    # print np.shape(good_old), np.shape(good_new), np.shape(u1), np.shape(v1), np.shape(u2), np.shape(v2)
    ux = (u2 - u1)/(x2 - x1)
    vx = (v2 - v1)/(x2 - x1)
    uy = (u2 - u1)/(y2 - y1)
    vy = (v2 - v1)/(y2 - y1)
    u1 = u2.copy()
    v1 = v2.copy()
    divergence = ux + vy
    defcos = ux - vy
    defsin = uy + vx
    deformation = np.sqrt(np.square(defcos) + np.square(defsin))
    divv = np.average(divergence)
    defv = np.average(deformation)
    print 2/divv
    p0 = p1
    # draw the tracks
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    #     cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    # img = cv2.add(frame, mask)
    # cv2.imshow('frame', img)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
cv2.destroyAllWindows()
cap.release()