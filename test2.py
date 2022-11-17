#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import time


import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:9872")


# settings camera C270
mtx = np.float32(
    [
        [794.71614391, 0.00000000e00, 347.55631962],
        [0.00000000e00, 794.71614391, 293.50160806],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

dist = np.float32(
    [
        [-2.45415937e-01],
        [-6.48440697e00],
        [3.54169640e-02],
        [9.11031500e-03],
        [-1.09181519e02],
        [-1.23188350e-01],
        [-7.76776901e00],
        [-1.05816513e02],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
        [0.00000000e00],
    ]
)

camMatrix = mtx
distCoeffs = dist

inputVideo = cv.VideoCapture(0)

inputVideo.set(
    cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G")
)  # depends on fourcc available camera

inputVideo.set(cv.CAP_PROP_FRAME_WIDTH, 320)
inputVideo.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
inputVideo.set(cv.CAP_PROP_FPS, 100)


# dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
# dictionary.bytesList = dictionary.bytesList[0]
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
# dictionary.bytesList = dictionary.bytesList[20]

# used to record the time when we processed last frame
prev_frame_time = time.time()
start_time = prev_frame_time
frames = 0


def arucoAnal(overlayPose=False):
    (corners, ids, impts) = cv.aruco.detectMarkers(image, dictionary)

    if ids is not None:

        cv.aruco.drawDetectedMarkers(imageCopy, corners, ids)

        (rvecs, tvecs, objpts) = cv.aruco.estimatePoseSingleMarkers(
            corners, 0.12, camMatrix, distCoeffs
        )

        # print(tvecs.shape)

        if overlayPose:
            for rvec, tvec in zip(rvecs, tvecs):
                # project axis points
                axisPoints = np.float32(
                    [
                        [0, 0, 0],
                        [0.12, 0, 0],
                        [0, 0.12, 0],
                        [0, 0, 0.12],
                    ]
                ).reshape((-1, 1, 3))

                (imagePoints, jacobian) = cv.projectPoints(
                    axisPoints, rvec, tvec, camMatrix, distCoeffs
                )
                imagePoints = imagePoints.astype(int)
                cv.line(imageCopy, imagePoints[0, 0], imagePoints[1, 0], (0, 0, 255), 3)
                cv.line(imageCopy, imagePoints[0, 0], imagePoints[2, 0], (0, 255, 0), 3)
                cv.line(imageCopy, imagePoints[0, 0], imagePoints[3, 0], (255, 0, 0), 3)


while True:
    # measure fps of the loop
    frames += 1
    inputVideo.grab()
    (rv, image) = inputVideo.retrieve()
    if not rv:
        break

    # resize image to width 176
    # image = cv.resize(image, (176, 144))
    # image = cv.resize(image, (64, 48))
    # imageCopy = image.copy()
    imageCopy = image
    arucoAnal(True)

    # Calculating the fps
    new_frame_time = time.time()
    delta_time = new_frame_time - start_time
    cur_fps = np.around(frames / delta_time, 1)

    data = {
        "timestamp": new_frame_time,
        "cur_fps": np.round(1 / (new_frame_time - prev_frame_time), 2),
        "fps": cur_fps,
    }
    socket.send_string(json.dumps(data))
    prev_frame_time = new_frame_time

    cv.imshow("out", imageCopy)
    key = cv.waitKey(1)
    if key == 27:
        break
