import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    contours,hierarchy = cv.findContours(mask, 1, 2)
    if len(contours):
        contours.sort(key= lambda x: cv.contourArea(x), reverse=True)
        cnt = contours[0]
        if cv.contourArea(cnt) < 1000:
            continue
        # print(cv.contourArea(cnt))
        M = cv.moments(cnt)
        # print( M )
        # Bitwise-AND mask and original image

        x,y,w,h = cv.boundingRect(cnt)
        # print("here")
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # rect = cv.minAreaRect(cnt)
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # cv.drawContours(frame,[box],0,(0,0,255),2)
        img = frame[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
        clt = KMeans(n_clusters=3) #cluster number
        clt.fit(img)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)

        plt.axis("off")
        plt.imshow(bar)
        # plt.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv.destroyAllWindows()
