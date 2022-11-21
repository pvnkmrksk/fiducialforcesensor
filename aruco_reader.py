import numpy as np
import cv2.aruco as aruco
import cv2 as cv2
import datetime
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

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    rots = np.array([x, y, z])
    rots = np.array([np.degrees(r) for r in rots])
    rots[0] = 180 - rots[0] % 360
    return rots


def main():
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)
    cap.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
    )  # depends on fourcc available camera

    # set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # set framerate
    cap.set(cv2.CAP_PROP_FPS, 100)

    # set exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, 20)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # set gain
    cap.set(cv2.CAP_PROP_GAIN, 40)
    cap.set(cv2.CAP_PROP_GAMMA, 160)

    # used to record the time when we processed last frame
    prev_frame_time = time.time()
    start_time = prev_frame_time
    frames = 0

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    aruco_dict.bytesList = aruco_dict.bytesList[20]
    aruco_params = aruco.DetectorParameters_create()

    rvecs = None
    tvecs = None
    avg_fps = 0

    # main loop: retrieves and displays a frame from the camera
    while True:
        frames += 1
        new_frame_time = time.time()

        # blocks until the entire frame is read
        success, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            (rvecs, tvecs, objpts) = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.004, camMatrix, distCoeffs
            )

            rotMat, jacob = cv2.Rodrigues(rvecs)
            rots = rotationMatrixToEulerAngles(rotMat)

            tvecs = tvecs[0][0]

            # print("rvecs: ", rvecs)
            # print("rots: ", rots)
            # print("tvecs: ", tvecs)

            try:
                data = {
                    "timestamp": new_frame_time,
                    "x": tvecs[0],
                    "y": tvecs[1],
                    "z": tvecs[2],
                    "roll": rots[0],
                    "pitch": rots[1],
                    "yaw": rots[2],
                    "avg": avg_fps,
                    "cur": np.round(1 / (new_frame_time - prev_frame_time), 2),
                }
                socket.send_string(json.dumps(data))

            except Exception as e:
                print(e)
                print("tvecs: ", tvecs)
                print("rots: ", rots)

        # compute fps: current_time - last_time
        delta_time = new_frame_time - start_time
        avg_fps = np.around(frames / delta_time, 1)
        prev_frame_time = new_frame_time

        cv2.imshow("webcam", img)
        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
