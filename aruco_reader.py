import numpy as np
import cv2.aruco as aruco
import cv2 as cv2
import datetime
import time
import zmq
import json
import threading
import argparse
from queue import Queue


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="ArUco marker detection and pose estimation",
        epilog="""
Examples:
  python aruco_reader.py --tag-size 0.05 --camera 1 --width 640 --height 480
  python aruco_reader.py --aruco-dict DICT_5X5_100 --port 5555
""",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.01,
        help="Size of the ArUco tag in meters (default: 0.01)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera width resolution (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=960,
        help="Camera height resolution (default: 960)",
    )
    parser.add_argument(
        "--fps", type=int, default=120, help="Camera FPS (default: 120)"
    )
    parser.add_argument(
        "--exposure", type=int, default=1, help="Camera exposure (default: 1)"
    )
    parser.add_argument("--gain", type=int, default=1, help="Camera gain (default: 1)")
    parser.add_argument(
        "--gamma", type=int, default=72, help="Camera gamma (default: 72)"
    )
    parser.add_argument(
        "--brightness", type=int, default=0, help="Camera brightness (default: 0)"
    )
    parser.add_argument(
        "--contrast", type=int, default=32, help="Camera contrast (default: 32)"
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_ARUCO_ORIGINAL",
        choices=[
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_4X4_250",
            "DICT_4X4_1000",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_5X5_250",
            "DICT_5X5_1000",
            "DICT_6X6_50",
            "DICT_6X6_100",
            "DICT_6X6_250",
            "DICT_6X6_1000",
            "DICT_7X7_50",
            "DICT_7X7_100",
            "DICT_7X7_250",
            "DICT_7X7_1000",
            "DICT_ARUCO_ORIGINAL",
        ],
        help="ArUco dictionary to use (default: DICT_ARUCO_ORIGINAL)",
    )
    parser.add_argument(
        "--subset-id",
        type=int,
        default=64,
        help="ArUco marker subset ID for DICT_ARUCO_ORIGINAL (default: 64)",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=500,
        help="Number of frames to use for baseline calculation (default: 500)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9872,
        help="ZMQ port for publishing pose data (default: 9872)",
    )
    parser.add_argument(
        "--min-pixel-size",
        type=int,
        default=250,
        help="Minimum marker side length in pixels (default: 100)",
    )
    parser.add_argument(
        "--max-pixel-size",
        type=int,
        default=700,
        help="Maximum marker side length in pixels (default: 700)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug visualization of rejected markers",
    )

    return parser.parse_args()


# Initialize ZMQ context - move this to main to use the port from args
def init_zmq(port):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    return socket


# read in camera matrix and distortion coefficients
with np.load("camera_calibration_results.npz") as X:
    camMatrix, distCoeffs, _, _ = [
        X[i] for i in ("camera_matrix", "dist_coeffs", "rvecs", "tvecs")
    ]

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
    camera=0,
    width=320,
    height=240,
    fps=100,
    exposure=150,
    gain=40,
    gamma=160,
    brightness=0,
    contrast=32,
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

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # set exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    # set gain and gamma
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_GAMMA, gamma)

    # set brightness
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

    # set contrast
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    return cap


def read_image(cap):
    # blocks until the entire frame is read
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def get_pose(
    img,
    gray,
    aruco_dict,
    aruco_params,
    tagSize,
    prev_marker_info=None,
    min_pixel_size=100,
    max_pixel_size=700,
    debug=False,
):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )

    if debug and rejectedImgPoints is not None and len(rejectedImgPoints) > 0:
        # Draw rejected markers in red
        aruco.drawDetectedMarkers(img, rejectedImgPoints, None, (0, 0, 255))

    if ids is not None:
        # Draw detected markers in green
        aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))

        # Calculate marker sizes and positions
        marker_info = []
        for i, corner in enumerate(corners):
            # Calculate marker side length in pixels
            x_coords = corner[0][:, 0]
            y_coords = corner[0][:, 1]

            # Calculate all four sides
            sides = []
            for j in range(4):
                next_j = (j + 1) % 4
                side_length = np.sqrt(
                    (x_coords[j] - x_coords[next_j]) ** 2
                    + (y_coords[j] - y_coords[next_j]) ** 2
                )
                sides.append(side_length)

            # Use minimum side length for filtering
            min_side = min(sides)
            avg_side = sum(sides) / 4

            # Calculate center position
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            marker_info.append(
                {
                    "id": ids[i][0],
                    "min_side": min_side,
                    "avg_side": avg_side,
                    "center": (center_x, center_y),
                    "corners": corner[0],
                    "index": i,
                }
            )

        # Filter markers by side length
        valid_markers = [
            m for m in marker_info if min_pixel_size <= m["min_side"] <= max_pixel_size
        ]

        if debug:
            # Draw size information for all detected markers
            for marker in marker_info:
                center = marker["center"]
                size_text = f"ID:{marker['id']} Min:{marker['min_side']:.1f} Avg:{marker['avg_side']:.1f}"
                cv2.putText(
                    img,
                    size_text,
                    (int(center[0]), int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        if not valid_markers:
            return [None, None, None], [None, None, None], None

        # If we have previous marker info, find the most similar marker
        selected_marker = None
        if prev_marker_info is not None:
            min_diff = float("inf")
            for marker in valid_markers:
                # Calculate difference in size and position
                size_diff = abs(marker["avg_side"] - prev_marker_info["avg_side"])
                pos_diff = np.sqrt(
                    (marker["center"][0] - prev_marker_info["center"][0]) ** 2
                    + (marker["center"][1] - prev_marker_info["center"][1]) ** 2
                )
                total_diff = size_diff + pos_diff

                if total_diff < min_diff:
                    min_diff = total_diff
                    selected_marker = marker
        else:
            # If no previous marker, use the one with the most consistent side lengths
            selected_marker = min(
                valid_markers,
                key=lambda x: max(x["min_side"], x["avg_side"])
                - min(x["min_side"], x["avg_side"]),
            )

        try:
            # Use the selected marker's corners for pose estimation
            marker_corners = np.array([selected_marker["corners"]])
            (rvecs, tvecs, objpts) = aruco.estimatePoseSingleMarkers(
                marker_corners, tagSize, camMatrix, distCoeffs
            )

            if rvecs is not None and len(rvecs) > 0:
                rvec = rvecs[0][0]
                if rvec.shape == (3,) or rvec.shape == (3, 1):
                    rotMat, jacob = cv2.Rodrigues(rvec)
                    rots = rotationMatrixToEulerAngles(rotMat)
                    tvecs = tvecs[0][0]
                    return rots, tvecs, selected_marker
        except Exception as e:
            print(f"Error in pose estimation: {e}")

    return [None, None, None], [None, None, None], None


def read_get_pose(
    img,
    gray,
    aruco_dict,
    aruco_params,
    rots_bl,
    tvecs_bl,
    tagSize,
    prev_marker_info=None,
    min_pixel_size=100,
    max_pixel_size=700,
    debug=False,
):
    rots, tvecs, marker_info = get_pose(
        img,
        gray,
        aruco_dict,
        aruco_params,
        tagSize,
        prev_marker_info,
        min_pixel_size,
        max_pixel_size,
        debug,
    )

    if rots[0] is not None and tvecs[0] is not None:
        rots = rots - rots_bl
        tvecs = tvecs - tvecs_bl
    else:
        rots = [None, None, None]
        tvecs = [None, None, None]

    return rots, tvecs, marker_info


def get_baseline(cap, aruco_dict, aruco_params, tagSize, frames=10, socket=None):
    rots = []
    tvecs = []

    rots_bl = np.array([0, 0, 0])
    tvecs_bl = np.array([0, 0, 0])
    prev_marker_info = None

    for i in range(frames):
        img, gray = read_image(cap)

        rots_i, tvecs_i, marker_info = read_get_pose(
            img,
            gray,
            aruco_dict,
            aruco_params,
            rots_bl,
            tvecs_bl,
            tagSize,
            prev_marker_info,
        )

        if rots_i[0] is not None and tvecs_i[0] is not None:
            rots.append(rots_i)
            tvecs.append(tvecs_i)
            prev_marker_info = marker_info

        cv2.imshow("webcam", img)
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
    # Parse command line arguments
    args = parse_args()

    # Print configuration
    print("=== ArUco Reader Configuration ===")
    print(f"Tag size: {args.tag_size}m")
    print(f"Camera ID: {args.camera}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"ArUco dictionary: {args.aruco_dict}")
    if args.subset_id is not None:
        print(f"Subset ID: {args.subset_id}")
    print(f"Marker size limits: {args.min_pixel_size}-{args.max_pixel_size} pixels")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Baseline frames: {args.baseline_frames}")
    print(f"ZMQ port: {args.port}")
    print("================================")

    # Initialize ZMQ socket
    socket = init_zmq(args.port)

    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)  # Adjust the number of threads based on your GPU

    tagSize = args.tag_size  # Get tag size from args

    # Initialize camera with parameters from args
    cap = initCamera(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        exposure=args.exposure,
        gain=args.gain,
        gamma=args.gamma,
        brightness=args.brightness,
        contrast=args.contrast,
    )

    # Verify camera configuration
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("\n=== Camera Configuration Verification ===")
    print(f"Requested resolution: {args.width}x{args.height}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    print(f"Requested FPS: {args.fps}")
    print(f"Actual FPS: {actual_fps}")
    print("=======================================\n")

    # Get the ArUco dictionary based on args
    dict_mapping = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    }

    dict_id = dict_mapping[args.aruco_dict]
    aruco_dict = aruco.Dictionary_get(dict_id)

    # Apply subset ID if specified, regardless of dictionary type
    if args.subset_id is not None:
        try:
            aruco_dict.bytesList = aruco_dict.bytesList[args.subset_id]
            print(f"Applied subset ID {args.subset_id} to dictionary")
        except Exception as e:
            print(f"Warning: Could not apply subset ID {args.subset_id}: {e}")
            print("Continuing with full dictionary...")

    aruco_params = aruco.DetectorParameters_create()

    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10

    rots_bl, tvecs_bl = get_baseline(
        cap,
        aruco_dict,
        aruco_params,
        tagSize,
        frames=args.baseline_frames,
        socket=socket,
    )

    avg_fps, cur_fps, frames = 0, 0, 0
    prev_frame_time = time.time()
    start_time = prev_frame_time

    length = 100
    rots_q = np.zeros((length, 3))
    tvecs_q = np.zeros((length, 3))

    time_header = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    frame_queue = Queue(maxsize=1)
    camera_io = threading.Thread(target=camera_io_thread, args=(cap, frame_queue))
    camera_io.daemon = True
    camera_io.start()

    prev_marker_info = None
    while True:
        frames += 1
        try:
            img, gray = frame_queue.get()
            rots, tvecs, marker_info = read_get_pose(
                img,
                gray,
                aruco_dict,
                aruco_params,
                rots_bl,
                tvecs_bl,
                tagSize,
                prev_marker_info,
                args.min_pixel_size,
                args.max_pixel_size,
                args.debug,
            )
            if marker_info is not None:
                prev_marker_info = marker_info
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
