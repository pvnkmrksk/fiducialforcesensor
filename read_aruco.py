# http://www.philipzucker.com/aruco-in-opencv/
'''
Run as sanity check that we can open the webcam and detect a tag (doesn't use threads).

Date: 19 Sept 2019
Author: nouyang

(Passing note: apparently cv2.aruco detection uses the apriltag algorithm
https://github.com/opencv/opencv_contrib/pull/1637)

This was used for debugging only, not for collecting data for the paper.

Usage:
$ python read_aruco.py
$ python read_aruco.py 2 # use if device number of webcam is not zero

numTags can also be set to change the number of tags expected in the frame. My initial prototype had one tag, and the final one has two tags.
'''

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import pprint

#---------- CHANGE DEVICE NUMBER AS NEEDED
if len(sys.argv) > 1:
    CAM_NUM = int(sys.argv[1])
    print('trying camera', CAM_NUM)
    cap = cv2.VideoCapture( CAM_NUM)
else:
    cap = cv2.VideoCapture(0)
# help(cv2.aruco)

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
# Note: The possibilites as per https://github.com/opencv/opencv_contrib/blob/96ea9a0d8a2dee4ec97ebdec8f79f3c8a24de3b0/modules/aruco/samples/create_board.cpp
# "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
# "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
# "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
# "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"

print(aruco_dict)

np.set_printoptions(suppress=True, precision=1)


# --------------------
# Camera matrix constants 

# NOTE: I do not have actual camera matrix for the Amazon hawk sensor! (original
# flat)
# Approximate using another webcam...  HBV-1716-2019-01-29_calib.yaml
# Note: Calibration calculated using https://github.com/smidm/video2calibration/

cameraMatrix = np.array([[521.92676671,   0.,         315.15287785], 
 [  0.,         519.01808261, 193.05763006],
 [  0.,           0.,           1.        ]])

distCoeffs, rvecs, tvecs = np.array([]), [], []

distCoeffs =  np.array([ 0.02207713,  0.18641578, -0.01917194, -0.01310851,
                        -0.11910311])

## ------------------------------
distBtwTags = 0.004 #center to center
tagSize = 0.0038
translate_vec = np.array([45, 0, 0])
euler_xyz_vec = np.array([45, 0, 0]) # in degrees

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2]) # roll
        y = math.atan2(-R[2,0], sy)  # pitch
        z = math.atan2(R[1,0], R[0,0]) # yaw
    else: # gimbal lock
        print('gimbal lock')
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    rots = np.array([x, y, z])
    rots = np.array([math.degrees(r) for r in rots])


    rots[0] = 180 - rots[0] % 360 


## ------------------------------

numTags = 2
out, prevOut = np.ones((numTags, 3)), np.ones((numTags, 3))
# w = 0.2 # 0.2 = fairly heavy weighting
WEIGHT = 0.1
# plt.axis([0,10,0,1])
count = 0
x,y = [0], [-45]
fig = plt.figure()

starttime, inittime = time.time(), time.time()
now = 0
elapsed = 0
rate = 0
plt.plot(count, y)
plt.title('Example Fiducial Force-Torque Sensor Interface')
plt.xlabel('Time (s)')
plt.ylabel('Y Readings (not real units)')

AXIZ = 2

rvec, tvec = None, None
fields = ['Fx: ', 'Fy: ', 'Fz: ', 'Mx: ', 'My: ', 'Mz: ']

readings = []
x_readings = []
y_readings = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read(0.)
    # print(frame.shape) #480x640

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    # Param defaults:
    # https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
    parameters = aruco.DetectorParameters_create()


    # Note: parameters can be set like so:
    # parameters.minMarkerPerimeterRate = 0.25


    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # print(corners)

    gray = aruco.drawDetectedMarkers(gray, corners)
    cv2.imshow('frame', gray)

    if (ids is not None) and (len(ids) == numTags):
        tagData = zip(ids, corners)
        tagData = sorted(tagData, key=lambda x: x[0]) # put corners in order of id
        ids = [tag[0] for tag in tagData]
        corners = [tag[1] for tag in tagData]
        # rvec, tvec, objPoints = \
        rvec, tvec = \
            cv2.aruco.estimatePoseSingleMarkers(corners, tagSize,
                                                cameraMatrix,
                                                distCoeffs);

    # convert rotMat to Euler, and smooth with filter 
    if rvec is not None and rvec.shape[0] == numTags:
        count += 1
        if (count % 2) == 0:
            now = time.time()
            elapsed = now - starttime
            starttime = now
            rate = 2 / elapsed
        for i in range(numTags):
            rotMat, jacob = cv2.Rodrigues(rvec[i].flatten())
            rots = rotationMatrixToEulerAngles(rotMat)

            out[i] = WEIGHT * rots + (1- WEIGHT) * prevOut[i]
            prevOut[i] = out[i]


        # print(rvec.shape)
        out = out.reshape((numTags,1,3))

        tagIdx = 1
        # print one side first
        reading = np.concatenate((tvec[tagIdx]*10000, rvec[tagIdx]*100)).flatten()
        # reading = np.concatenate((tvec[1]*10000, out[1])).flatten()
        reading = np.round(reading, decimals=3)
        readings.append(reading)
        a = ['{0: <10}'.format(x) for x in reading]
        # pprint.pprint(''.join([val for pair in zip(fields, a) for val in \
                               # pair]))
        # print(count)
        print(reading[AXIZ])
        #plt.plot(count, out.flatten()[AXIZ], 'k.')
        x_readings.append(count)
        y_readings.append(reading[AXIZ])
        plt.gca().lines[0].set_xdata(x_readings);
        plt.gca().lines[0].set_ydata(y_readings);
        plt.gca().relim();
        plt.gca().autoscale_view();
        plt.pause(0.001);

    # print(rejectedImgPoints)
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(np.array(readings).std(axis=0)*1000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        endtime = time.time()
        break

# When everything done, release the capture
print('Program init time', inittime, 'endtime', endtime, 'counts', count)
print('Counts/sec (Hz)', count/(endtime - inittime))
cap.release()
cv2.destroyAllWindows()


