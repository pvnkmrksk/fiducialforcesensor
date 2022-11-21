import numpy as np
import cv2
import datetime
import cv2.aruco as aruco

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


def main():
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)
    cap.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
    )  # depends on fourcc available camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 50)
    # retrieve properties of the capture object
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    fps_sleep = int(1000 / cap_fps)
    print("* Capture width:", cap_width)
    print("* Capture height:", cap_height)
    print("* Capture FPS:", cap_fps, "ideal wait time between frames:", fps_sleep, "ms")

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0
    # used to record the time when we processed last frame
    prev_frame_time = time.time()
    start_time = prev_frame_time
    frames = 0

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    aruco_dict.bytesList = aruco_dict.bytesList[20]

    print(type(aruco_dict))
    # print the aruco dictionary

    aruco_params = aruco.DetectorParameters_create()

    # main loop: retrieves and displays a frame from the camera
    while True:
        frames += 1

        # blocks until the entire frame is read
        success, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            (rvecs, tvecs, objpts) = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.12, camMatrix, distCoeffs
            )

        new_frame_time = time.time()

        # compute fps: current_time - last_time
        delta_time = new_frame_time - start_time
        cur_fps = np.around(frames / delta_time, 1)

        cv2.imshow("webcam", img)
        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

        data = {
            "timestamp": new_frame_time,
            "cur_fps": np.round(1 / (new_frame_time - prev_frame_time), 2),
            "fps": cur_fps,
        }
        socket.send_string(json.dumps(data))
        prev_frame_time = new_frame_time

    # release resources
    cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":

    main()
