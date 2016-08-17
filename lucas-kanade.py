import numpy as np
import cv2
from scipy import ndimage

def translationlk(template, new_image, p0):
    win1, win2 = 5, 5
    w1 = int(win1/2)
    w2 = int(win2/2)
    p1 = np.zeros_like(p0)
    st = np.ones(p0.shape[0])
    for i, x in enumerate(p0):
        a, b = x.ravel()
        xwin = np.arange(a - w1, a + w1 + 1)
        ywin = np.arange(b - w2, b + w2 + 1)
        xwin = np.array([xwin] * win1)
        ywin = np.array([ywin] * win2).transpose()
        #gx, gy = np.gradient(new_image)
        gx = cv2.Sobel(new_image, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(new_image, cv2.CV_64F, 0, 1, ksize=5)
        t_mask = template[b-w2:b+w2+1][:, a-w1:a+w1+1]
        p = np.zeros((2, 1))
        dp = np.ones((2, 1))
        Wp = np.array([[1, 0], [0, 1]])
        j = 0
        while abs(dp[0]) > 0.1 or abs(dp[1]) > 0.1:
            xwarp = xwin + p[0]*np.ones_like(xwin)
            ywarp = ywin + p[1]*np.ones_like(ywin)
            arr = np.array([ywarp, xwarp])
            i_warp = ndimage.map_coordinates(new_image, arr)
            error = t_mask - i_warp
            gx_warp = ndimage.map_coordinates(gx, arr)
            gy_warp = ndimage.map_coordinates(gy, arr)
            steep_x = np.kron(Wp[0], gx_warp)
            steep_y = np.kron(Wp[1], gy_warp)
            steep = steep_x + steep_y
            hes = np.dot(steep.T, steep)
            temp = np.array([hes[:win1, :win2], hes[:win1, win2:], hes[win1:, :win2], hes[win1:, win2:]])
            hessian = temp.sum(1).sum(1).reshape(2, 2)
            if hessian.shape[0] == hessian.shape[1] and np.linalg.matrix_rank(hessian) == hessian.shape[0]:
                inv_hessian = np.linalg.inv(hessian)
            else:
                st[i] = 0
                break
            temp = np.array([steep[:, :win1], steep[:, win2:]])
            temp = np.dot(temp, error)
            sd_update = temp.sum(1).sum(1).reshape(2, 1)
            dp = np.dot(inv_hessian, sd_update)
            p = p + dp
            j = j+1
            if j > 50:
                st[i] = 0
                break

        p1[i] = np.array([int(a+p[0]), int(b+p[1])])
    return p1, st

def affinelk(template, new_image, p0):
    win1, win2 = 9, 9
    w1 = int(win1/2)
    w2 = int(win2/2)
    p1 = np.zeros_like(p0)
    st = np.ones(p0.shape[0])
    for i, x in enumerate(p0):
        a, b = x.ravel()
        xwin = np.arange(a - w1, a + w1 + 1)
        ywin = np.arange(b - w2, b + w2 + 1)
        xwin = np.array([xwin] * win1)
        ywin = np.array([ywin] * win2).transpose()
        #gx, gy = np.gradient(new_image)
        gx = cv2.Sobel(new_image, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(new_image, cv2.CV_64F, 0, 1, ksize=5)
        t_mask = template[b-w2:b+w2+1][:, a-w1:a+w1+1]
        p = np.zeros((6, 1))
        dp = np.ones((6, 1))
        Wp = np.array([[a, 0, b, 0, 1, 0], [0, a, 0, b, 0, 1]])
        pts2 = np.float32([[xwin[0, 0], ywin[0, 0]], [xwin[1, 1], ywin[1, 1]], [xwin[2, 2], ywin[2, 2]]])
        rows, cols = new_image.shape
        j = 0
        while all(entry > 0.1 for entry in abs(dp)):
            xwarp = (1+p[0])*xwin + p[2]*ywin + p[4]*np.ones_like(xwin)
            ywarp = p[1]*xwin + (1+p[3])*ywin + p[5]*np.ones_like(ywin)
            # pts1 = np.float32([[xwarp[0, 0], ywarp[0, 0]], [xwarp[1, 1], ywarp[1, 1]], [xwarp[2, 2], ywarp[2, 2]]])
            # M = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
            #M = cv2.getAffineTransform(pts1, pts2)
            i_warp1 = cv2.warpAffine(new_image, M, (cols, rows))
            # cv2.imshow('dsd', i_warp1)
            # cv2.waitKey(0)
            arr = np.array([ywarp, xwarp])
            i_warp = ndimage.map_coordinates(new_image, arr)
            # cv2.imshow('dsd', i_warp)
            # cv2.waitKey(0)
            error = t_mask - i_warp
            gx_warp = ndimage.map_coordinates(gx, arr)
            gy_warp = ndimage.map_coordinates(gy, arr)
            # steep_x = np.kron(Wp[0], gx_warp)
            # steep_y = np.kron(Wp[1], gy_warp)
            # steep = steep_x + steep_y
            # hes = np.split(steep, 6, axis=1)
            steep = np.array([xwin*gx_warp, xwin*gy_warp, ywin*gx_warp, ywin*gy_warp, gx_warp, gy_warp])
            hessian = np.zeros((6, 6))
            for k in range(6):
                for l in range(6):
                    hessian[k, l] = np.dot(steep[k].T, steep[l]).sum()
            if hessian.shape[0] == hessian.shape[1] and np.linalg.matrix_rank(hessian) == hessian.shape[0]:
                inv_hessian = np.linalg.inv(hessian)
            else:
                st[i] = 0
                break
            temp = np.dot(steep, error)
            sd_update = temp.sum(1).sum(1).reshape(6, 1)
            dp = np.dot(inv_hessian, sd_update)
            p = p + dp
            j = j+1
            if j > 50:
                st[i] = 0
                break

        p1[i] = np.array([int(a+p[0]), int(b+p[1])])
    return p1, st


img1 = cv2.imread('img1.1.jpg')
img2 = cv2.imread('trans.jpg')
img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
color = np.random.randint(0, 255, (100, 3))
feature_params = dict(maxCorners=20, qualityLevel=0.1, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(img1_grey, mask=None, **feature_params)
mask = np.zeros_like(img1)
frame = img1
p1, st = affinelk(img1_grey, img2_grey, p0)
good_new = p1[st == 1]
good_old = p0[st == 1]
    # draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    frame = cv2.circle(frame, (a, b), 1, color[i].tolist(), -1)
img = cv2.add(frame, mask)
cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
