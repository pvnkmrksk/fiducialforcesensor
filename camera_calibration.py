import cv2
import cv2.aruco as aruco
import numpy as np
from aruco_reader import initCamera

# Define the ArUco dictionary and parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_dict.bytesList = np.array([aruco_dict.bytesList[64]])  # Select the 20th marker

parameters = aruco.DetectorParameters_create()

# Prepare for calibration
all_corners = []
all_ids = []
img_points = []
obj_points = []

calibration_flags = (
    cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
)

# ArUco tag's size (the length of a side, specified by you in meters)
aruco_square_dimension = 0.01  # Replace with your marker's side length

# Define the 3D points of the ArUco marker
objp = np.zeros((4, 3), dtype=np.float32)
objp[1, 0] = aruco_square_dimension
objp[2, :] = [aruco_square_dimension, aruco_square_dimension, 0]
objp[3, 1] = aruco_square_dimension


# You can use your webcam, a video file or a set of images to collect the samples
cap = initCamera(
    camera=0, width=640, height=480, fps=100, exposure=22, gain=15, gamma=72
)
frame_counter = 0
calibrate_every_n_frames = 10
while True:
    ret, frame = cap.read()
    frame_counter += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Display the ArUco markers detected in the image
        frame = aruco.drawDetectedMarkers(frame, corners)

        # Append every Nth frame
        if frame_counter % calibrate_every_n_frames == 0:
            all_corners.append(corners)
            all_ids.append(ids)

            for corner in corners:
                img_points.append(corner)
                obj_points.append(objp)

    cv2.imshow("frame", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# After collecting enough samples, perform the calibration

# save all the samples needed for calibration and have an option to calibrate from them
np.savez(
    "camera_calibration.npz",
    all_corners=all_corners,
    all_ids=all_ids,
    img_points=img_points,
    obj_points=obj_points,
)

if False:
    # load the samples
    npzfile = np.load("calibration.npz")
    all_corners = npzfile["all_corners"]
    all_ids = npzfile["all_ids"]
    img_points = npzfile["img_points"]
    obj_points = npzfile["obj_points"]


retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None, flags=calibration_flags
)

print("Camera Matrix: ", camera_matrix)
print("Distortion Coefficients: ", dist_coeffs)

# Save the calibration results to a file
np.savez(
    "camera_calibration_results.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs,
)
