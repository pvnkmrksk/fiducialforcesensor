import cv2
import numpy as np

# Load your image
img = cv2.imread("/home/lab/my_photo-2.jpg")

# Your camera matrix and distortion coefficients from the calibration
# Your camera matrix and distortion coefficients from the calibration
camera_matrix = np.array(
    [
        [3.18178993e03, 0.00000000e00, 3.56480383e02],
        [0.00000000e00, 3.21028738e03, 2.68906759e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
dist_coeffs = np.array([-9.14669792, -0.43721955, 0.0, 0.0, 0.0])

# Undistort the image
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

# Display the original and undistorted images
cv2.imshow("Original Image", img)
cv2.imshow("Undistorted Image", undistorted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
