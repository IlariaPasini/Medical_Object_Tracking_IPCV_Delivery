import cv2
import numpy as np
from matplotlib import pyplot as plt

k = np.ones((7,7),np.float32)/49
path = 'test_images/solidWhiteCurve.jpg'

src = cv2.imread(path)

gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,500,apertureSize = 3)

smoothed = cv2.filter2D(src,-1, kernel=k)

gray_s = cv2.cvtColor(smoothed,cv2.COLOR_BGR2GRAY)
edges_s = cv2.Canny(gray_s, 100, 170,apertureSize = 3)

#dilate = cv2.dilate(edges, np.ones((2,2)), iterations=1)
lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=10,minLineLength=50,maxLineGap=10)
lines_s = cv2.HoughLinesP(edges_s,1,np.pi/180,threshold=5,minLineLength=10,maxLineGap=20)


for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(src,(x1,y1),(x2,y2),(0,255,0),1)

for line in lines_s:
    x1,y1,x2,y2 = line[0]
    cv2.line(smoothed,(x1,y1),(x2,y2),(0,255,0),1)

# visualizzation

plt.subplot(221),plt.imshow(edges)
plt.title("edges"),plt.axis("off")
plt.subplot(222),plt.imshow(src)
plt.title("result"),plt.axis("off")
plt.subplot(223),plt.imshow(edges_s)
plt.title("edges"),plt.axis("off")
plt.subplot(224),plt.imshow(smoothed)
plt.title("smoothed"),plt.axis("off")

plt.show()