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

# # settings camera C270
# mtx = np.float32(
#     [
#         [794.71614391, 0.00000000e00, 347.55631962],
#         [0.00000000e00, 794.71614391, 293.50160806],
#         [0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )

# dist = np.float32(
#     [
#         [-2.45415937e-01],
#         [-6.48440697e00],
#         [3.54169640e-02],
#         [9.11031500e-03],
#         [-1.09181519e02],
#         [-1.23188350e-01],
#         [-7.76776901e00],
#         [-1.05816513e02],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#     ]
# )

# camMatrix = mtx
# distCoeffs = dist


camMatrix = np.array(
    [
        [3.18178993e03, 0.00000000e00, 3.56480383e02],
        [0.00000000e00, 3.21028738e03, 2.68906759e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
distCoeffs = np.array([-9.14669792, -0.43721955, 0.0, 0.0, 0.0])


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


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


def initCamera(
    camera=0, width=320, height=240, fps=100, exposure=150, gain=40, gamma=160
):
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(camera)
    cap.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
    )  # depends on fourcc available camera

    # set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # set fps
    cap.set(cv2.CAP_PROP_FPS, fps)

    # set exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # set gain and gamma
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_GAMMA, gamma)

    return cap


def read_image(cap):
    # blocks until the entire frame is read
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def get_pose(img, gray, aruco_dict, aruco_params):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )

    if ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)

        (rvecs, tvecs, objpts) = aruco.estimatePoseSingleMarkers(
            corners, 0.004, camMatrix, distCoeffs
        )
        rotMat, jacob = cv2.Rodrigues(rvecs)
        rots = rotationMatrixToEulerAngles(rotMat)
        tvecs = tvecs[0][0]

    else:
        tvecs = [None, None, None]
        rots = [None, None, None]

    return rots, tvecs


def read_get_pose(cap, aruco_dict, aruco_params, rots_bl, tvecs_bl):
    img, gray = read_image(cap)
    rots, tvecs = get_pose(img, gray, aruco_dict, aruco_params)

    if rots[0] is not None and tvecs[0] is not None:
        rots = rots - rots_bl
        tvecs = tvecs - tvecs_bl

    else:
        rots = [None, None, None]
        tvecs = [None, None, None]

    return rots, tvecs, img


def get_baseline(cap, aruco_dict, aruco_params, frames=10):
    rots = []
    tvecs = []

    rots_bl = np.array([0, 0, 0])
    tvecs_bl = np.array([0, 0, 0])

    for i in range(frames):
        rots_i, tvecs_i, img = read_get_pose(
            cap, aruco_dict, aruco_params, rots_bl, tvecs_bl
        )

        if rots_i[0] is not None and tvecs_i[0] is not None:
            rots.append(rots_i)
            tvecs.append(tvecs_i)

        cv2.imshow("webcam", img)
        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        send_pose(socket, rots_i, tvecs_i, 0, 0)

    rots_bl = np.array(rots).mean(axis=0)
    tvecs_bl = np.array(tvecs).mean(axis=0)

    send_pose(socket, rots_bl, tvecs_bl, 0, 0)

    return rots_bl, tvecs_bl


def send_pose(socket, rots, tvecs, avg_fps, cur_fps, raw=None):
    if raw is None:
        raw = [0, 0, 0]
    data = {
        "x": tvecs[0],
        "y": tvecs[1],
        "z": tvecs[2],
        "roll": rots[0],
        "pitch": rots[1],
        "yaw": rots[2],
        # "raw0": raw[0],
        # "raw1": raw[1],
        # "raw2": raw[2],
        # "avg": avg_fps,
        # "cur": cur_fps,
    }
    # only send data if all values are not None
    if not any(np.array(list(data.values())) == None):
        socket.send_json(data)


def med_filter(q, data, length=11, threshold=3):
    """
    This function performs a median filter on data. It takes in a queue, data,
    length, and threshold. The length is the size of the queue and the threshold
    is the z score threshold for which to replace the data in the queue with the
    last element in the queue. The function returns the median of the queue.

    Parameters:
    q: Queue
    data: Number
    length: Number
    threshold: Number

    Returns:
    Number: the median of the queue
    """

    # if any of the items in the data is None, replace data with last element in queue
    if not any(np.array(data) == None):
        np.roll(q, -1, axis=0)

        std = np.std(q, axis=0)
        # if any of the std is 0, then replace data with last element in queue
        if std.any() == 0:
            q[-1] = data
            print(f"std is 0, replacing data with {data}")
            return q, np.median(q[-length:], axis=0)

        # # get z score of q and data and replace data if z score is greater than 3
        # z = (data - np.mean(q, axis=0)) / std
        # # if any of the z scores are greater than 3, replace data with last element in queue
        # if any(abs(z) > threshold):
        #     data = q[-1]
        #     print(f"z score {z} is greater than {threshold}, replacing data with {data}")

        q[-1] = data

    return q, np.median(q[-length:], axis=0)


def main():
    # initialize camera
    cap = initCamera(
        camera=0, width=320, height=240, fps=100, exposure=10, gain=1, gamma=72
    )

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    aruco_dict.bytesList = aruco_dict.bytesList[64]
    aruco_params = aruco.DetectorParameters_create()

    rots_bl, tvecs_bl = get_baseline(cap, aruco_dict, aruco_params, frames=100)

    # used to record the time when we processed last frame
    avg_fps, cur_fps, frames = 0, 0, 0
    prev_frame_time = time.time()
    start_time = prev_frame_time

    length = 100
    # define a deque for rots and tvecs
    rots_q = np.zeros(
        (length, 3),
    )
    tvecs_q = np.zeros(
        (length, 3),
    )

    # main loop: retrieves and displays a frame from the camera
    while True:
        frames += 1
        new_frame_time = time.time()

        rots, tvecs, img = read_get_pose(
            cap, aruco_dict, aruco_params, rots_bl, tvecs_bl
        )

        raw = rots.copy()
        # smooth out the rots and tvecs data
        # rots_q, raw = med_filter(rots_q, rots, length=11, threshold=50)

        # rots_q = np.roll(rots_q, -1, axis=0)
        # rots_q[-1] = rots
        # raw = np.median(rots_q[-7:], axis=0)
        # raw = np.median([[1,2,3],[1,2,3],[1,2,3],[4,5,6]], axis=0)

        send_pose(socket, rots, tvecs, avg_fps, cur_fps, raw=raw)

        # compute fps: current_time - last_time
        delta_time = new_frame_time - start_time
        avg_fps = np.around(frames / delta_time, 1)
        cur_fps = np.round(1 / (new_frame_time - prev_frame_time), 2)
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
