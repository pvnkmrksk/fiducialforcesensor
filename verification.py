import numpy as np
import cv2
from aruco_reader import initCamera, camMatrix, distCoeffs


def undistort_image(img, cam_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
        cam_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted_img = cv2.undistort(img, cam_matrix, dist_coeffs, None, new_cam_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y : y + h, x : x + w]

    return cv2.resize(
        undistorted_img, (w, h)
    )  # Resize undistorted image to match raw frame size


def sbs_raw_undistorted_viewer(camera=0):
    cap = initCamera(
        camera=0, width=320, height=240, fps=100, exposure=10, gain=1, gamma=72
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_raw = frame.copy()
        frame_undistorted = undistort_image(frame, camMatrix, distCoeffs)

        frame_undistorted = cv2.resize(
            frame_undistorted, (frame_raw.shape[1], frame_raw.shape[0])
        )

        frame_sbs = np.concatenate((frame_raw, frame_undistorted), axis=1)

        cv2.imshow("webcam", frame_sbs)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


sbs_raw_undistorted_viewer()
