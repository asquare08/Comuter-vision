import numpy as np
import cv2
from scipy import ndimage


def lucasKanadeICAffine(template, new_image, p0):
    win1, win2 = 9, 9
    w1 = int(win1 / 2)
    w2 = int(win2 / 2)
    p1 = np.zeros_like(p0)
    st = np.ones(p0.shape[0])
    for i, x in enumerate(p0):
        a, b = x.ravel()
        xwin = np.arange(a - w1, a + w1 + 1)
        ywin = np.arange(b - w2, b + w2 + 1)
        xwin = np.array([xwin] * win1)
        ywin = np.array([ywin] * win2).transpose()
        # gx, gy = np.gradient(new_image)
        gx = cv2.Sobel(new_image, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(new_image, cv2.CV_64F, 0, 1, ksize=5)
        t_mask = template[b - w2:b + w2 + 1][:, a - w1:a + w1 + 1]
        p = np.zeros((6, 1))
        dp = np.ones((6, 1))
        Wp = np.array([[a, 0, b, 0, 1, 0], [0, a, 0, b, 0, 1]])
        pts2 = np.float32([[xwin[0, 0], ywin[0, 0]], [xwin[1, 1], ywin[1, 1]], [xwin[2, 2], ywin[2, 2]]])
        rows, cols = new_image.shape
        j = 0
        while all(entry > 0.1 for entry in abs(dp)):
            xwarp = (1 + p[0]) * xwin + p[2] * ywin + p[4] * np.ones_like(xwin)
            ywarp = p[1] * xwin + (1 + p[3]) * ywin + p[5] * np.ones_like(ywin)
            # pts1 = np.float32([[xwarp[0, 0], ywarp[0, 0]], [xwarp[1, 1], ywarp[1, 1]], [xwarp[2, 2], ywarp[2, 2]]])
            # M = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
            # M = cv2.getAffineTransform(pts1, pts2)
            # i_warp1 = cv2.warpAffine(new_image, M, (cols, rows))
            # cv2.imshow('dsd', i_warp1)
            # cv2.waitKey(0)
            arr = np.array([ywarp, xwarp])
            i_warp = ndimage.map_coordinates(new_image, arr)
            #cv2.imshow('dsd', i_warp)
            #cv2.waitKey(0)
            if t_mask.shape != i_warp.shape:
                continue
            error = t_mask - i_warp
            gx_warp = ndimage.map_coordinates(gx, arr)
            gy_warp = ndimage.map_coordinates(gy, arr)
            # steep_x = np.kron(Wp[0], gx_warp)
            # steep_y = np.kron(Wp[1], gy_warp)
            # steep = steep_x + steep_y
            # hes = np.split(steep, 6, axis=1)
            steep = np.array([xwin * gx_warp, xwin * gy_warp, ywin * gx_warp, ywin * gy_warp, gx_warp, gy_warp])
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
            j = j + 1
            if j > 50:
                st[i] = 0
                break

        p1[i] = np.array([int(a + p[0]), int(b + p[1])])
    return p1, st


def lucasKanadeICTranslational(template, new_image, p0):
    win1, win2 = 15, 15
    w1 = int(win1 / 2)
    w2 = int(win2 / 2)
    p1 = np.zeros_like(p0)
    st = np.ones(p0.shape[0])
    # gx, gy = np.gradient(template)
    gx = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=5)
    for i, x in enumerate(p0):
        a, b = x.ravel()
        xwin = np.arange(a - w1, a + w1 + 1)
        ywin = np.arange(b - w2, b + w2 + 1)
        xwin = np.array([xwin] * win1)
        ywin = np.array([ywin] * win2).transpose()
        t_mask = template[b - w2:b + w2 + 1][:, a - w1:a + w1 + 1]
        gx_mask = gx[b - w2:b + w2 + 1][:, a - w1:a + w1 + 1]
        gy_mask = gy[b - w2:b + w2 + 1][:, a - w1:a + w1 + 1]
        p = np.zeros((2, 1))
        dp = np.ones((2, 1))
        Wp = np.array([[1, 0], [0, 1]])
        steep_x = np.kron(Wp[0], gx_mask)
        steep_y = np.kron(Wp[1], gy_mask)
        steep = steep_x + steep_y
        hes = np.dot(steep.T, steep)
        temp = np.array([hes[:win1, :win2], hes[:win1, win2:], hes[win1:, :win2], hes[win1:, win2:]])
        hessian = temp.sum(1).sum(1).reshape(2, 2)
        if hessian.shape[0] == hessian.shape[1] and np.linalg.matrix_rank(hessian) == hessian.shape[0]:
            inv_hessian = np.linalg.inv(hessian)
        else:
            st[i] = 0
            continue
        j = 0
        xwarp = xwin
        ywarp = ywin
        while abs(dp[0]) > 0.2 or abs(dp[1]) > 0.2:
            arr = np.array([ywarp, xwarp])
            i_warp = ndimage.map_coordinates(new_image, arr)
            error = i_warp - t_mask
            temp = np.array([steep[:, :win1], steep[:, win2:]])
            temp = np.dot(temp, error)
            sd_update = temp.sum(1).sum(1).reshape(2, 1)
            dp = np.dot(inv_hessian, sd_update)
            # dp_new = np.array([-dp[0]-dp[0]*dp[3]+dp[1]*dp[2], -dp[1], -dp[2], -dp[3]-dp[0]*dp[3]+dp[1]*dp[2], -dp[4]-dp[3]*dp[4]+dp[2]*dp[5], -dp[5]-dp[0]*dp[5]+dp[1]*dp[4]])
            # dp_new = dp_new / (((1+dp[0])*(1+dp[3]))-(dp[1]*dp[2]))
            # xtemp = (1+dp_new[0])*xwarp + dp_new[2]*ywarp + dp_new[4]*np.ones_like(xwarp)
            # ytemp = dp_new[1]*xwarp + (1+dp_new[3])*ywarp + dp_new[5]*np.ones_like(ywarp)
            # xwarp = (1 + p[0])*xtemp + p[2]*ytemp + p[4]*np.ones_like(xtemp)
            # ywarp = p[1]*xtemp + (1 + p[3])*ytemp + p[5]*np.ones_like(ytemp)
            p = p - dp
            xwarp = xwarp - dp[0] * np.ones_like(xwarp)
            ywarp = ywarp - dp[1] * np.ones_like(ywarp)
            j = j + 1
            if j > 50:
                st[i] = 0
                break
        p1[i] = np.array([int(a + p[0]), int(b + p[1])])
    return p1, st
