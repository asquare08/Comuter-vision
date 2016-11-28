import numpy as np
import cv2
from scipy.spatial import distance


class LicencePlateDetector:
    def __init__(self):
        self.i = 0

    def detect(self, frame_gray):
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        blur = cv2.blur(frame_gray, (7,7))
        edges = cv2.Canny(blur, 250, 500, apertureSize=5)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, (0,0,255), thickness=1)

        kids = []
        dist = []
        i = 0
        for cnt1 in contours:
            kids.append(0)
            dist.append(0)
            x, y, w, h = cv2.boundingRect(cnt1)
            if w <= h or w*h < 500 or h*1.5 > w:
                i += 1
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            j = 0
            for cnt2 in contours:
                x2, y2, w2, h2 = cv2.boundingRect(cnt2)
                if x2 in range(x, x+w) and y2 in range(y, y+h) and w2*h2 > 20:
                    kids[i] += 1
                    dist[i] += distance.euclidean((x,y),(x2,y2))
                j += 1
            cv2.putText(frame, str(kids[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            i += 1

        if kids:
            i = 0
            for kid in kids:
                if kid < 2*np.mean(kids) or kid < 17:
                    dist[i] = 10000000000000000000
                i += 1
            index = np.argmin(dist)
            if dist[index] < 1000000000:
                return cv2.boundingRect(contours[index])

        self.i += 1
        return None
