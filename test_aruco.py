import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return
        
    # Set camera properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_GAIN, 1)
    cap.set(cv2.CAP_PROP_GAMMA, 72)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    
    # Load camera calibration
    try:
        with np.load("camera_calibration_results.npz") as X:
            camMatrix, distCoeffs, _, _ = [
                X[i] for i in ("camera_matrix", "dist_coeffs", "rvecs", "tvecs")
            ]
    except Exception as e:
        print(f"Error loading camera calibration: {e}")
        print("Using default camera matrix")
        camMatrix = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float32)
        distCoeffs = np.zeros((5, 1), dtype=np.float32)
    
    # Use original dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = aruco.DetectorParameters_create()
    
    # Adjust parameters to be more lenient
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.minMarkerPerimeterRate = 0.03
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.03
    aruco_params.minCornerDistanceRate = 0.05
    aruco_params.minDistanceToBorder = 3
    aruco_params.minOtsuStdDev = 5.0
    aruco_params.perspectiveRemovePixelPerCell = 4
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    aruco_params.maxErroneousBitsInBorderRate = 0.35
    aruco_params.minOtsuStdDev = 5.0
    aruco_params.errorCorrectionRate = 0.6
    
    # Marker size in meters
    marker_size = 0.01  # 1cm
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Basic image processing
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply some basic processing
        # 1. Normalize brightness
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # 2. Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # 3. Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
        
        # Show the processed grayscale image
        cv2.imshow('Processed', gray)
        
        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        
        # Draw all potential markers
        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))  # Green for detected
            print(f"Detected marker IDs: {ids}")
            
            # Estimate pose for detected markers
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camMatrix, distCoeffs)
            
            # Draw axis for each detected marker
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camMatrix, distCoeffs, rvecs[i], tvecs[i], marker_size/2)
        
        if rejected is not None:
            aruco.drawDetectedMarkers(frame, rejected, None, (0, 0, 255))  # Red for rejected
            print(f"Rejected markers: {len(rejected)}")
        
        # Display original frame with markers
        cv2.imshow('ArUco Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 