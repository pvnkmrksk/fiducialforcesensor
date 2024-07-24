import cv2
import numpy as np
import click
import os
from aruco_reader import initCamera

# Define the ArUco dictionary and parameters
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
ARUCO_DICT.bytesList = ARUCO_DICT.bytesList[64]
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

def capture_images(cap, num_images, decimation_factor, output_dir):
    images = []
    frame_count = 0
    while len(images) < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Display the live feed
        cv2.imshow('Live Feed', frame)
        
        # Capture every nth frame
        if frame_count % decimation_factor == 0:
            img_path = os.path.join(output_dir, f"image_{len(images)}.jpg")
            cv2.imwrite(img_path, frame)
            images.append(img_path)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return images

def calibrate_camera(image_paths, aruco_dict, aruco_params, marker_length):
    all_corners = []
    all_ids = []
    img_shape = None

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if img_shape is None:
            img_shape = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is not None:
            all_corners.append(corners)
            all_ids.append(ids)

    if not all_corners:
        raise ValueError("No markers detected. Ensure the images contain the specified ArUco marker.")

    obj_points = []
    img_points = []
    obj_p = np.zeros((4, 3), np.float32)
    obj_p[:, :2] = np.array([[0, 0], [marker_length, 0], [marker_length, marker_length], [0, marker_length]])

    for corners in all_corners:
        for corner in corners:
            img_points.append(corner)
            obj_points.append(obj_p)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape[::-1], None, None)

    return camera_matrix, dist_coeffs

def undistort_image(image_path, camera_matrix, dist_coeffs):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_img, new_camera_matrix, dist_coeffs

@click.command()
@click.option('--num-images', '-n', type=int, default=100, help='Number of images to capture for calibration.')
@click.option('--marker-length', '-ml', type=float, default=0.1, help='Length of the ArUco marker side in the same units as calibration (e.g., meters).')
@click.option('--iterations', '-it', type=int, default=5, help='Number of iterations for self-calibration.')
@click.option('--output-dir', '-o', default='calibration_images', help='Directory to save captured images and calibration results.')
@click.option('--decimation-factor', '-df', type=int, default=25, help='Decimation factor for capturing every nth frame.')
def main(num_images, marker_length, iterations, output_dir, decimation_factor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = initCamera(camera=0, width=1280, height=960, fps=120, exposure=10, gain=10, gamma=72)
    images = capture_images(cap, num_images, decimation_factor, output_dir)

    camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        camera_matrix, dist_coeffs = calibrate_camera(images, ARUCO_DICT, ARUCO_PARAMS, marker_length)

        # Save interim results
        with open(f'{output_dir}/calibration_iteration_{i+1}.yml', 'w') as f:
            f.write(f'camera_matrix: {camera_matrix.tolist()}\n')
            f.write(f'dist_coeffs: {dist_coeffs.tolist()}\n')

        # Undistort images for the next iteration
        undistorted_images = []
        for image_path in images:
            undistorted_img, camera_matrix, dist_coeffs = undistort_image(image_path, camera_matrix, dist_coeffs)
            undistorted_image_path = os.path.join(output_dir, f'undistorted_{i+1}_{os.path.basename(image_path)}')
            cv2.imwrite(undistorted_image_path, undistorted_img)
            undistorted_images.append(undistorted_image_path)
        images = undistorted_images

    # Save final results
    with open(f'{output_dir}/calibrated_camera.yml', 'w') as f:
        f.write(f'camera_matrix: {camera_matrix.tolist()}\n')
        f.write(f'dist_coeffs: {dist_coeffs.tolist()}\n')

    print(f"Calibration completed. Results saved to {output_dir}/calibrated_camera.yml")

if __name__ == '__main__':
    main()
