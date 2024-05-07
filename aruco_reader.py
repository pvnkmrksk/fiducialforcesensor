import numpy as np
import cv2.aruco as aruco
import cv2 as cv2
import datetime
import time
import zmq
import json
import threading
from queue import Queue

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:9872")

# read in camera matrix and distortion coefficients
with np.load('camera_calibration_results.npz') as X:
    camMatrix, distCoeffs, _, _ = [X[i] for i in ('camera_matrix', 'dist_coeffs', 'rvecs', 'tvecs')]

# ... (keep the utility functions as is)



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
    camera=0, width=320, height=240, fps=100, exposure=150, gain=40, gamma=160, brightness=0, contrast=32
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


    #set brightness
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

    #set contrast
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    return cap


def read_image(cap):
    # blocks until the entire frame is read
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def get_pose(img, gray, aruco_dict, aruco_params, tagSize):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )

    if ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)

        (rvecs, tvecs, objpts) = aruco.estimatePoseSingleMarkers(
            corners, tagSize, camMatrix, distCoeffs
        )
        rotMat, jacob = cv2.Rodrigues(rvecs)
        rots = rotationMatrixToEulerAngles(rotMat)
        tvecs = tvecs[0][0]

    else:
        tvecs = [None, None, None]
        rots = [None, None, None]

    return rots, tvecs
def read_get_pose(img, gray, aruco_dict, aruco_params, rots_bl, tvecs_bl, tagSize):
    rots, tvecs = get_pose(img, gray, aruco_dict, aruco_params, tagSize)

    if rots[0] is not None and tvecs[0] is not None:
        rots = rots - rots_bl
        tvecs = tvecs - tvecs_bl
    else:
        rots = [None, None, None]
        tvecs = [None, None, None]

    return rots, tvecs


def get_baseline(cap, aruco_dict, aruco_params, tagSize,frames=10):
    rots = []
    tvecs = []

    rots_bl = np.array([0, 0, 0])
    tvecs_bl = np.array([0, 0, 0])

    for i in range(frames):
        img, gray = read_image(cap)

        rots_i, tvecs_i = read_get_pose(
            img, gray, aruco_dict, aruco_params, rots_bl, tvecs_bl, tagSize=0.01
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

    try:
        data = {
            "x": tvecs[0] if tvecs[0] is not None else np.nan,
            "y": tvecs[1] if tvecs[1] is not None else np.nan,
            "z": tvecs[2] if tvecs[2] is not None else np.nan,
            "roll": rots[0] if rots[0] is not None else np.nan,
            "pitch": rots[1] if rots[1] is not None else np.nan,
            "yaw": rots[2] if rots[2] is not None else np.nan,
        }
    except Exception as e:
        print(e)
        return

    # Send data even if some values are None
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




def camera_io_thread(cap, frame_queue):
    while True:
        try:
            img, gray = read_image(cap)
            frame_queue.put((img, gray))
        except Exception as e:
            print(e)
            

def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)  # Adjust the number of threads based on your GPU

    tagSize = 0.01  # units in meters. tvecs Output is in meters
    # cap = initCamera(camera=0, width=640, height=480, fps=120, exposure=22, gain=12, gamma=72)
    # cap = initCamera(camera=0, width=1280, height=960, fps=120, exposure=22, gain=12, gamma=72)
    cap = initCamera(camera=0, width=1280, height=960, fps=120, exposure=10, gain=10, gamma=72)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    aruco_dict.bytesList = aruco_dict.bytesList[64]
    aruco_params = aruco.DetectorParameters_create()

    rots_bl, tvecs_bl = get_baseline(cap, aruco_dict, aruco_params, tagSize, frames=500)

    avg_fps, cur_fps, frames = 0, 0, 0
    prev_frame_time = time.time()
    start_time = prev_frame_time

    length = 100
    rots_q = np.zeros((length, 3))
    tvecs_q = np.zeros((length, 3))

    time_header = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    frame_queue = Queue(maxsize=1)
    camera_io = threading.Thread(target=camera_io_thread, args=(cap, frame_queue))
    camera_io.daemon = True
    camera_io.start()

    while True:
        frames += 1
        try:
            img, gray = frame_queue.get()
            rots, tvecs = read_get_pose(img, gray, aruco_dict, aruco_params, rots_bl, tvecs_bl, tagSize)
        except Exception as e:
            print(e)
            continue

        raw = rots.copy()
        send_pose(socket, rots, tvecs, avg_fps, cur_fps, raw=raw)

        cv2.imshow("webcam", img)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# ... (keep the profiling code as is)
import cProfile
import pstats

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        pass
# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     try:
#         main()

#     except KeyboardInterrupt:
#         pass
#     profiler.disable()
#     stats = pstats.Stats(profiler)
#     stats.dump_stats('profile_results.prof')


