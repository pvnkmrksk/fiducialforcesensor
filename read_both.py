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

import json

# import serial
import sys
import signal
import logging
import pprint

# import optoforcelibrary as optoforce
import copy


from datetime import datetime
import time

import threading
from argparse import ArgumentParser

import math
import numpy as np
import logging


#! /usr/local/bin/python3
import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:9872")

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
writeFlag = False  # True  # write to CSV file?

# ------------------------------------
# Data recording constants

# writeFolder = "./data/"
writeFolder = ""
strtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# a_fname = writeFolder + strtime + '_arucotagData.csv'
a_fname = strtime + "_arucotagData.csv"
fmode = "a"
inittime = time.time()


# ------------------------------------
# Camera Constants
# NOTE: Make sure to change this to match your prototype!
tag_size = 0.0038  # in meters

width = 320
height = 240
fps = 100

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

        self.aruco_dict.bytesList = self.aruco_dict.bytesList[20]

        self.aruco_params = aruco.DetectorParameters_create()

        # ------------------------------------
        # self.w = 0.2  # filter_weight
        self.w = 1.0  # Don't filter.
        # ------------------------------------
        # Open video stream
        try:
            # self.stream = cv2.VideoCapture(int(options.device_or_movie))
            self.stream = cv2.VideoCapture(0)
            self.stream.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
            )  # depends on fourcc available camera
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_FPS, fps)
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.stream.set(cv2.CAP_PROP_EXPOSURE, 20)
            self.stream.set(cv2.CAP_PROP_GAIN, 40)
            self.stream.set(cv2.CAP_PROP_GAMMA, 160)

        except ValueError:
            print("exception")
            self.stream = cv2.VideoCapture(options.device_or_movie)

    def run(self):
        # -----------------------------------
        # Set up filtering for angles
        # We use a straightforward average
        cumTvec, cumRvec = np.ones((numTags, 3)), np.ones((numTags, 3))
        # -----------------------------------
        # Initialized zeros
        counter = 0
        avgN = 10

        # used to record the time when we processed last frame
        prev_frame_time = 0
        new_frame_time = 0

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

        # used to record the time when we processed last frame
        prev_frame_time = time.time()
        start_time = prev_frame_time
        frames = 0

        while True:
            frames += 1
            success, frame = self.stream.read()

            if not success:
                self.logger.debug("failed to grab aruco frame")
                print("failed to grab aruco frame")

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

            if rvec is not None and rvec.shape[0] == numTags:
                tvec = tvec.reshape(numTags, 3)
                # NOTE: HARDCODED
                ids_str = str([str(id) for id in ids])
                # self.logger.info("ids: " + ids_str + " tvec: " + str(tvec.flatten()))
                atime = time.time()
                for i in range(numTags):
                    rotMat, jacob = cv2.Rodrigues(rvec[i])
                    rots = rotationMatrixToEulerAngles(rotMat)

                    # NOTE: EXPONENTIAL FILTERING
                    out[i] = self.w * rots + (1 - self.w) * prevOut[i]
                    prevOut[i] = out[i]

                # collect zeros at the beginning
                # x,y,z should be zero'd and then changes averaged between the
                # two tags (maybe?)

                # NOTE: we are exponential filtering thetas but not distances
                calcDists = tvec - zeroDists
                calcThetas = out - zeroThetas
                # message = f"{calcDists[0][0]} {calcDists[0][1]} {calcDists[0][2]} {calcThetas[0][0]} {calcThetas[0][1]} {calcThetas[0][2]}"

                # socket.send_string(json.dumps(data))

                new_frame_time = time.time()

                # compute fps: current_time - last_time
                total_time = new_frame_time - start_time
                avg_fps = np.around(frames / total_time, 1)

                # data = {
                #     "timestamp": new_frame_time,
                #     "cur_fps": np.round(1 / (new_frame_time - prev_frame_time), 2),
                #     "fps": cur_fps,
                # }

                data = {
                    "timestamp": new_frame_time,
                    "x": calcDists[0][0],
                    "y": calcDists[0][1],
                    "z": calcDists[0][2],
                    "roll": calcThetas[0][0],
                    "pitch": calcThetas[0][1],
                    "yaw": calcThetas[0][2],
                    "avg": avg_fps,
                    "cur": np.round(1 / (new_frame_time - prev_frame_time), 2),
                }
                socket.send_string(json.dumps(data))
                prev_frame_time = new_frame_time

                cv2.imshow("webcam", gray)
                # wait 1ms for ESC to be pressed
                key = cv2.waitKey(1)
                if key == 27:
                    break

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


# ------------------------------------
def main(options):

    inittime = time.time()
    cv2.namedWindow("Aruco Camera")

    global o_fname, a_fname
    if options.name is not None:
        a_fname = writeFolder + options.name + a_fname
    else:
        a_fname = writeFolder + a_fname

    if arucoFlag:
        # arucot = ArucoThread(inittime, camera_matrix, a_fname)
        arucot = ArucoThread(inittime, a_fname, cameraMatrix, distCoeffs, tag_size)
        arucot.daemon = True
        arucot.start()

    # cv2.startWindowThread()

    while threading.active_count() > 0:

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
