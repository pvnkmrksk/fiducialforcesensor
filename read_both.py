"""
Uses threads to simultaneously:
Read arucotag using openCV library.
and Read optoforce, using library from ShadowHand.

Date: 16 Sept 2019
Author: nouyang

With code from https://github.com/shadow-robot/optoforce/blob/indigo-devel/optoforce/src/optoforce/optoforce_node.py
Tip: ctrl-backslash for extra serious force quit (or sudo killall python)
NOTE: may require running `sudo chmod 666 /dev/ttyACM0` due to permissions
issues



Usage:  
$ python read_both.py --name='torque_calib'
or, if the camera is not found, try different numbers, e.g.
$ python read_both.py 2

A cv2 window should pop up, with the webcam view in grayscale. Wait a second for the "zero" readings to be taken. (For more info, see the video complementing the paper at https://sites.google.com/view/fiducialforcesensor).

Note: The "--name" is prefixed to output files.

Output:
Three files, a .jpg screenshot of the intiial view will be saved, as well as two data files from the aruco and opto sensors respectively.

"""

import cv2
import cv2.aruco as aruco

import serial
import sys
import signal
import logging
import pprint

import optoforcelibrary as optoforce
import copy


from datetime import datetime
import time

import threading
from argparse import ArgumentParser

import math
import numpy as np
import logging


# ------------------------------------
# Run Flags

# np.set_printoptions(suppress=True, precision=4)

# Use flags to indicate whether
# For instance, if window is frozen / no window is showing for the webcam,
# the arucotag thread may have failed and you only see output for the opto
# In that case disable the opto and only run the aruco to see the error messages
arucoFlag = True
numTags = 1
optoFlag = False
writeFlag = True  # write to CSV file?

# ------------------------------------
# Data recording constants

# writeFolder = "./data/"
writeFolder = ""
strtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# o_fname = writeFolder + strtime + '_optoforceData.csv'
# a_fname = writeFolder + strtime + '_arucotagData.csv'
o_fname = strtime + "_optoforceData.csv"
a_fname = strtime + "_arucotagData.csv"
fmode = "a"

inittime = time.time()

# -------------

# ------------------------------------
# Sensor Constants
optoforce_port = "/dev/ttyACM0"

# ------------------------------------
# Camera Constants
# NOTE: Make sure to change this to match your prototype!
tag_size = 0.0038 # in meters

width = 640
height = 480
fx = 640.0  # must be floats for apriltags._draw_pose to work
fy = 640.0
px = 300.0  # assume principal point to be center of image
py = 250.0

# camera_matrix = np.array([fx, fy, px, py])

# cameraMatrix = np.array([[619.6207333698724, 0.0, 283.87165814439834],
# [0.0, 613.2842389650563, 231.9993874728696],
# [0.0, 0.0, 1.0]])


# NOTE: I do not have actual camera matrix for the Amazon hawk sensor!
# Approximate using another webcam...  HBV-1716-2019-01-29_calib.yaml
# Use https://github.com/smidm/video2calibration/ and print out the checkerboard to calibrate your camera.
# 1. Print out checkerboard 2. Take video using webcam of waving / rotating checkerboard (don't move too fast)
# 3. $ mkdir out; ./calibrate.py your_checkerboard_video.mp4 calibration.yaml --debug-dir out
# Then copy the matrix and distortion coefficients below:

cameraMatrix = np.array(
    [
        [521.92676671, 0.0, 315.15287785],
        [0.0, 519.01808261, 193.05763006],
        [0.0, 0.0, 1.0],
    ]
)

distCoeffs, rvecs, tvecs = np.array([]), [], []

distCoeffs = np.array([0.02207713, 0.18641578, -0.01917194, -0.01310851, -0.11910311])

# ------------------------------------
# Methods to read aruco tag data


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    rots = np.array([x, y, z])
    rots = np.array([math.degrees(r) for r in rots])

    rots[0] = 180 - rots[0] % 360
    return rots


# ------------------------------------
# Methods to read from optoforce


class OptoThread(threading.Thread):
    def __init__(self, inittime, fname):

        threading.Thread.__init__(self)

        global logger
        self.logger = logger

        self.inittime = inittime
        self.fname = fname

        optoforce_sensor_type = "s-ch/6-axis"
        starting_index = 0
        scaling_factors = [[1, 1, 1, 1, 1, 1]]  # one per axis
        # if len(sys.argv) > 1:
        # port = "/dev/" + sys.argv[1]
        try:
            self.driver = optoforce.OptoforceDriver(
                optoforce_port, optoforce_sensor_type, scaling_factors
            )
            self.logger.info("Opened optoforce")
        except serial.SerialException as e:
            self.logger.debug("failed to open optoforce serial port!")
            raise

    def run(self):
        time.sleep(2)  # wait for opencv tag to initialize
        while True:
            optotime = time.time()
            self.optodata = self.driver.read()

            if isinstance(self.optodata, optoforce.OptoforceData):  # UNTESTED
                opto_data_str = (
                    "Optoforce time, xyz, yawpitchroll; "
                    + str(optotime - self.inittime)
                    + "; "
                    + "; ".join([str(o) for o in self.optodata.force[0]])
                    + "\n"
                )
                a = ["{0: <8}".format(x) for x in self.optodata.force[0]]
                # pprint.pprint(' '.join(a))  # NOTE

                # print(opto_data_str)
                if writeFlag:
                    with open(self.fname, "a") as outf:
                        outf.write(opto_data_str)
                        outf.flush()

            elif isinstance(data, optoforce.OptoforceSerialNumber):
                self.logger.info("The sensor's serial number is " + str(data))


# ------------------------------------
# Methods to read from arucotags


class ArucoThread(threading.Thread):
    def __init__(self, inittime, fname, camera_matrix, dist_coeffs, tag_size):
        threading.Thread.__init__(self)

        global logger
        self.logger = logger

        self.inittime = inittime
        self.fname = fname
        # self.camera_matrix = camera_matrix
        self.cameraMatrix = camera_matrix
        self.tagSize = tag_size
        self.distCoeffs = dist_coeffs

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters_create()

        # width=1280
        # height=720
        width = 320
        height = 240
        fps = 60

        # ------------------------------------
        self.w = 0.2  # filter_weight
        # self.w = 1.0 # Don't filter.

        # ------------------------------------
        # Parse commandline arguments

        # ------------------------------------
        # Open video stream
        try:
            # self.stream = cv2.VideoCapture(int(options.device_or_movie))
            self.stream = cv2.VideoCapture(0)
            self.stream.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
            )  # depends on fourcc available camera
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.stream.set(cv2.CAP_PROP_FPS, 120)
            # self.stream.set(cv2.CAP_PROP_GAIN, 100)
            # self.stream.set(cv2.CAP_PROP_EXPOSURE, -8.0)

        except ValueError:
            self.stream = cv2.VideoCapture(options.device_or_movie)

        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        # self.stream.set(cv2.CAP_PROP_FPS,fps)

    def run(self):
        # -----------------------------------
        # Set up filtering for angles
        # We use a straightforward average
        cumTvec, cumRvec = np.ones((numTags, 3)), np.ones((numTags, 3))

        # -----------------------------------
        # Initialized zeros
        counter = 0
        avgN = 1

        # -- NOTE: this code is highly repetitive, refactor at some point
        while counter < avgN:
            self.logger.info("looking for 20 tags... found: " + str(counter))
            success, frame = self.stream.read(0.0)
            if not success:
                self.logger.debug("failed to grab aruco frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

            if (ids is not None) and (len(ids) == numTags):
                tagData = zip(ids, corners)
                tagData = sorted(
                    tagData, key=lambda x: x[0]
                )  # put corners in order of id
                ids = [tag[0] for tag in tagData]
                corners = [tag[1] for tag in tagData]
            # print(ids)

            print(
                cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.tagSize, self.cameraMatrix, self.distCoeffs
                )
            )
            # rvec, tvec = \

            rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.tagSize, self.cameraMatrix, self.distCoeffs
            )
            # print(rvec)

            if rvec is not None and rvec.shape[0] == numTags:
                counter += 1
                tvec = tvec.reshape(numTags, 3)

                cumTvec += tvec
                for i in range(numTags):
                    rotMat, jacob = cv2.Rodrigues(rvec[i])
                    rots = rotationMatrixToEulerAngles(rotMat)
                    cumRvec[i] = rots

        if writeFlag:
            cv2.imwrite(self.fname + "_.jpg", frame)
            # for debugging, save a frame
            if numTags == 1:
                data_str = "Arucotag time, x1,y1,z1,roll1,pitch1,yaw1,x2,y2,z2,roll2,pitch2,yaw2,expW\n"
            elif numTags == 2:
                data_str = "Arucotag time, x1,y1,z1,roll1,pitch1,yaw1,x2,y2,z2,roll2,pitch2,yaw2,x,y,z,roll,pitch,yaw,expW\n"
            else:
                print("What tags are you trying to do?")
                raise Exception

            with open(self.fname, "a") as outf:
                outf.write(data_str)
                outf.flush()

        zeroThetas = copy.deepcopy(cumRvec / avgN)
        zeroDists = copy.deepcopy(cumTvec / avgN)

        # ------------------------------------
        # Look for arucotags
        # out, prevOut = np.ones((numTags, 3)), np.ones((numTags, 3))
        out, prevOut = copy.deepcopy(zeroThetas), copy.deepcopy(zeroDists)

        while True:
            atime = time.time()
            success, frame = self.stream.read()

            if not success:
                self.logger.debug("failed to grab aruco frame")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

            # print('while loop ids', ids)

            rvec = None

            if (ids is not None) and (len(ids) == numTags):
                tagData = zip(ids, corners)
                tagData = sorted(
                    tagData, key=lambda x: x[0]
                )  # put corners in order of id
                ids = [tag[0] for tag in tagData]
                corners = [tag[1] for tag in tagData]

                # rvec, tvec = \

                rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.tagSize, self.cameraMatrix, self.distCoeffs
                )
                gray = aruco.drawDetectedMarkers(gray, corners)
                cv2.imshow("Aruco Camera", gray)

            if rvec is not None and rvec.shape[0] == numTags:
                tvec = tvec.reshape(numTags, 3)
                # NOTE: HARDCODED
                ids_str = str([str(id) for id in ids])
                self.logger.info("ids: " + ids_str + " tvec: " + str(tvec.flatten()))
                atime = time.time()
                for i in range(numTags):
                    rotMat, jacob = cv2.Rodrigues(rvec[i])
                    rots = rotationMatrixToEulerAngles(rotMat)

                    # NOTE: EXPONENTIAL FILTERING
                    out[i] = self.w * rots + (1 - self.w) * prevOut[i]
                    prevOut[i] = out[i]
                    # print('Aruco Tag ID:', i, ': ', out[i])
                    # print('Aruco Tag ID:', 1, ': ', out[1])

                # collect zeros at the beginning
                # x,y,z should be zero'd and then changes averaged between the
                # two tags (maybe?)

                # NOTE: we are exponential filtering thetas but not distances
                calcDists = tvec - zeroDists
                calcThetas = out - zeroThetas
                # Average between the two tags
                avgCalcDists = np.average(calcDists, axis=0)
                avgCalcThetas = np.average(calcThetas, axis=0)

                # print(tvec[1]* 1000)

                # print('shapes', [x.shape for x in [zeroThetas, out, calcThetas,
                # avgCalcThetas]])
                # print('zeroThetas', zeroThetas)
                # print('zeroDists', zeroDists)
                # print('filtered reading\n', out)
                self.logger.debug("exponential filtered, euler angles thetas")
                # print('zerod reading', calcThetas)
                # print('averaged between two', avgCalcThetas)
                # print('\n')
                # print(outD)
                # print(zeroDists)

                if writeFlag:
                    data_str = (
                        # "Arucotag time, tag1 xyz, tag2 xyz, "
                        # + "tag1 rollpitchyaw (xyz), tag2 -- xyz zerod averaged -- expo weight; "
                        str(atime - self.inittime)
                        + ", "
                        + ", ".join([str(t) for t in calcDists.flatten()])
                        + ", "
                        + ", ".join([str(r) for r in calcThetas.flatten()])
                        + ", "
                        + ", ".join([str(t) for t in avgCalcDists.flatten()])
                        + ", "
                        + ", ".join([str(r) for r in avgCalcThetas.flatten()])
                        + ","
                        + str(self.w)
                        + " \n"
                    )
                    #                     data_str = (
                    #     "Arucotag time, tag1 xyz, tag2 xyz, "
                    #     + "tag1 rollpitchyaw (xyz), tag2 -- xyz zerod averaged -- expo weight; "
                    #     + str(atime - self.inittime)
                    #     + "; "
                    #     + "; ".join([str(t) for t in calcDists.flatten()])
                    #     + "; "
                    #     + "; ".join([str(r) for r in calcThetas.flatten()])
                    #     + "; "
                    #     + "; ".join([str(t) for t in avgCalcDists.flatten()])
                    #     + "; "
                    #     + "; ".join([str(r) for r in avgCalcThetas.flatten()])
                    #     + ";"
                    #     + str(self.w)
                    #     + " \n"
                    # )
                    # '; '.join([str(t) for t in tvec.flatten()]) + '; ' + \
                    # '; '.join([str(r) for r in out.flatten()]) + '; '+ \

                    with open(self.fname, "a") as outf:
                        outf.write(data_str)
                        outf.flush()

                # a = ['{0: <8}'.format(x) for x in self.optodata.force[0]]
                # pprint.pprint(' '.join(a))


# ------------------------------------
def main(options):

    inittime = time.time()
    cv2.namedWindow("Aruco Camera")

    global o_fname, a_fname
    if options.name is not None:
        o_fname = writeFolder + options.name + o_fname
        a_fname = writeFolder + options.name + a_fname
    else:
        o_fname = writeFolder + o_fname
        a_fname = writeFolder + a_fname

    if optoFlag:
        optot = OptoThread(inittime, o_fname)
        optot.daemon = True
        optot.start()
    if arucoFlag:
        # arucot = ArucoThread(inittime, camera_matrix, a_fname)
        arucot = ArucoThread(inittime, a_fname, cameraMatrix, distCoeffs, tag_size)
        arucot.daemon = True
        arucot.start()

    # cv2.startWindowThread()

    while threading.active_count() > 0:
        # cv2.imshow("Camera", _overlay)
        # cv2.waitKey(0) #IMPORTANT, without this line, window hangs

        k = cv2.waitKey(0)
        if k == 27 or k == ord("q"):  # Escape key
            # self.stopped = True
            print("!! ---- Caught escape key")
            break
        time.sleep(0.001)

        # outf.close()
    print("!! ---- Destroying windows")
    print("=============== \nData STARTED at\n" + strtime)
    print("---------------\nData RECORDED to: ", a_fname)
    endtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("---------------\nData collection ENDED at", endtime, "\n==================")
    cv2.destroyAllWindows()
    sys.exit(0)


# ------------------------------------
if __name__ == "__main__":
    # print('=============== \nData STARTED at', strtime, '\n================')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("read_sensor")
    logger.info("Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def signal_handler(sig, frame):
        print("You pressed Ctrl+C!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = ArgumentParser()

    parser.add_argument(
        "device_or_movie",
        metavar="INPUT",
        nargs="?",
        default=0,
        help="Movie to load or integer ID of camera device",
    )
    parser.add_argument("--name", dest="name", default=None, help="name")
    options = parser.parse_args()

    try:
        main(options)
    except KeyboardInterrupt:
        print("!! --- Keyboard interrupt in progress")
    finally:
        print("!! --- Finished cleaning up.")
