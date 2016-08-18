import numpy as np
import cv2
from scipy import ndimage

def translationlk(template, new_image, p0):
    win1, win2 = 5, 5
    w1 = int(win1/2)
    w2 = int(win2/2)
    p1 = np.zeros_like(p0)
    st = np.ones(p0.shape[0])
    rows, cols = new_image.shape
    for i, x in enumerate(p0):
        a, b = x.ravel()
        xwin = np.arange(a - w1, a + w1 + 1)
        ywin = np.arange(b - w2, b + w2 + 1)
        xwin = np.array([xwin] * win1)
        ywin = np.array([ywin] * win2).transpose()
        # gx, gy = np.gradient(new_image)
        gx = cv2.Sobel(new_image, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(new_image, cv2.CV_64F, 0, 1, ksize=5)
        t_mask = template[b-w2:b+w2+1][:, a-w1:a+w1+1]
        p = np.zeros((2, 1))
        if t_mask.shape == (win1, win2):
            dp = np.ones((2, 1))
            Wp = np.array([[1, 0], [0, 1]])
            j = 0
            while abs(dp[0]) > 0.1 or abs(dp[1]) > 0.1:
                xwarp = xwin + p[0]*np.ones_like(xwin)
                ywarp = ywin + p[1]*np.ones_like(ywin)
                arr = np.array([ywarp, xwarp])
                M = np.float32([[1, 0, p[0]], [0, 1, p[1]]])
                i_warp_full = cv2.warpAffine(new_image, M, (cols, rows))
                i_warp = ndimage.map_coordinates(i_warp_full, arr)
                error = t_mask - i_warp
                gx_warp_full = cv2.warpAffine(gx, M, (cols, rows))
                gy_warp_full = cv2.warpAffine(gy, M, (cols, rows))
                gx_warp = ndimage.map_coordinates(gx_warp_full, arr)
                gy_warp = ndimage.map_coordinates(gy_warp_full, arr)
                steep = np.array([gx_warp, gy_warp])
                hessian = np.zeros((2, 2))
                for k in range(2):
                    for l in range(2):
                        hessian[k, l] = np.dot(steep[k].T, steep[l]).sum()
                if hessian.shape[0] == hessian.shape[1] and np.linalg.matrix_rank(hessian) == hessian.shape[0]:
                    inv_hessian = np.linalg.inv(hessian)
                else:
                    st[i] = 0
                    break
                sd_update = np.array([[np.dot(gx_warp.T, error).sum()], [np.dot(gy_warp.T, error).sum()]])
                dp = np.dot(inv_hessian, sd_update)
                p = p + dp
                j += 1
                if j > 50:
                    st[i] = 0
                    break
        p1[i] = np.array([int(a+p[0]), int(b+p[1])])
    return p1, st

def affinelk(template, new_image, p0):
    win1, win2 = 15, 15
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
        gx = cv2.Sobel(new_image, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(new_image, cv2.CV_64F, 0, 1, ksize=5)
        t_mask = template[b-w2:b+w2+1][:, a-w1:a+w1+1]
        p = np.zeros((6, 1))
        if t_mask.shape == (win1, win2):
            dp = np.ones((6, 1))
            rows, cols = new_image.shape
            j = 0
            while all(entry > 0.7 for entry in abs(dp)):
                xwarp = (1+p[0])*xwin + p[2]*ywin + p[4]*np.ones_like(xwin)
                ywarp = p[1]*xwin + (1+p[3])*ywin + p[5]*np.ones_like(ywin)
                M = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
                i_warp_full = cv2.warpAffine(new_image, M, (cols, rows))
                arr = np.array([ywarp, xwarp])
                i_warp = ndimage.map_coordinates(i_warp_full, arr)
                error = t_mask - i_warp
                gx_warp_full = cv2.warpAffine(gx, M, (cols, rows))
                gy_warp_full = cv2.warpAffine(gy, M, (cols, rows))
                gx_warp = ndimage.map_coordinates(gx_warp_full, arr)
                gy_warp = ndimage.map_coordinates(gy_warp_full, arr)
                steep = np.array([xwin*gx_warp, xwin*gy_warp, ywin*gx_warp, ywin*gy_warp, gx_warp, gy_warp])
                hessian = np.zeros((6, 6))
                sd_update = np.zeros((6, 1))
                for k in range(6):
                    for l in range(6):
                        hessian[k, l] = np.dot(steep[k].T, steep[l]).sum()
                    sd_update[k] = np.dot(steep[k].T, error).sum()
                if hessian.shape[0] == hessian.shape[1] and np.linalg.matrix_rank(hessian) == hessian.shape[0]:
                    inv_hessian = np.linalg.inv(hessian)
                else:
                    st[i] = 0
                    break
                dp = np.dot(inv_hessian, sd_update)
                p = p + dp
                j = j+1
                if j > 50:
                    st[i] = 0
                    break

        p1[i] = np.array([int(a+p[0]), int(b+p[1])])
    return p1, st

