import numpy as np
import cv2
from matplotlib import pyplot as plt

ROTATE_RF = 2
ROTATE_LF = 0

rf_path = "../samples/rf/rf_frame75.png"
imgRight = cv2.imread(rf_path, cv2.COLOR_BGR2GRAY)
#rotate the image in order to have the same orientation of the lf image
rf_rot = cv2.rotate(imgRight, ROTATE_RF)


lf_path = "../samples/lf/lf_frame75.png"
imgLeft = cv2.imread(lf_path, cv2.COLOR_BGR2GRAY)
#rotate the image in order to have the same orientation of the rf image
lf_rot = cv2.rotate(imgLeft, ROTATE_LF)


#rf_res = cv2.cvtColor(rf_rot, cv2.COLOR_BGR2RGB)
#lf_res = cv2.cvtColor(lf_rot, cv2.COLOR_BGR2RGB)

#show the original images in gray scale
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(lf_rot, 'gray')
plt.axis('on')
plt.subplot(1,2,2)
plt.imshow(rf_rot, 'gray')
plt.axis('on')
plt.show()

#all the parameters used in the stereoSGBM algorithm and used for the disparity map in stereoSGBM method
# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 11
min_disp = -128
max_disp = 128
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)

# Compute the disparity image
disparity = stereo.compute(lf_rot, rf_rot)

# Normalize the image for representation
min = disparity.min()
max = disparity.max()
# The normalization is store in another variable because we want to show the disparity map without normalization
disparity3 = np.uint8(255 * (disparity - min) / (max - min))



# Initialize the stereo block matching object 
bSize1=11
# Call the metohd in order to compute the disparity map with the StereoBM algorithm
stereo1 = cv2.StereoBM_create(numDisparities=16, blockSize=bSize1)

# Compute the disparity image
disparity1 = stereo1.compute(lf_rot, rf_rot)
min = disparity1.min()
max = disparity1.max()
# The normalization is store in another variable because we want to show the disparity map without normalization
disparity2 = np.uint8(255 * (disparity1 - min) / (max - min))


# Show the result for the first algorithm
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(disparity, 'gray')
plt.axis('on')
plt.title("Disparity map")
plt.subplot(1,2,2)
plt.imshow(disparity3, 'gray')
plt.axis('on')
plt.title("Normalized disparity map")
plt.show()

# Show the result for the second algorithm
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(disparity1, 'gray')
plt.axis('on')
plt.title("Disparity map")
plt.subplot(1,2,2)
plt.imshow(disparity2, 'gray')
plt.axis('on')
plt.title("Normalized disparity map")
plt.show()



# 4.87 mm +/- 5%. In addition to the 5% variation due to manufacturing tolerance, 
# #the focal length will change dynamically due to the autofocus system. The AF travel (stroke) is up to 0.2 mm.

#focal = ; we don't know the focal length of the camera 
#dimensionOfSensor = ; we don't know the dimension of the sensor
#distanceBetweenCameras = ; we don't know the distance between the cameras
#disparity; we get that from the disparity map
#map_profondita = (focal)/(dimensionOfSensor) * (distanceBetweenCameras) /(disparity)
#plt.imshow(map_profondita, cmap='gray') 
#plt.colorbar() 
#plt.show()