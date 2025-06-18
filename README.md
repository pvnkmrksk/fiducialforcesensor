## Introduction
# Fiducial Force Sensor

This project implements a low-cost, vision-based 6-axis force-torque sensor using ArUco fiducial markers and a standard camera. The approach tracks the 3D pose of a marker attached to a compliant structure, inferring forces and torques from observed displacements. The system leverages OpenCV's ArUco module for robust marker detection and pose estimation. Calibration routines are provided for both the camera and the flexure structure. Real-time pose data is published via ZMQ for integration with robotics and data acquisition systems.

## Installation

1. Install [uv](https://github.com/astral-sh/uv) (Python package installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Dependencies
- numpy
- opencv-contrib-python (for ArUco support)
- pyzmq
- click

If you encounter issues with `cv2.aruco`, ensure you have `opencv-contrib-python` installed (not just `opencv-python`).

## Usage

### Camera Calibration (optional, but recommended for best accuracy)
```bash
python camera_calibration.py
```
This will guide you through capturing calibration images and saving calibration data.

### Run ArUco Marker Detection
```bash
python aruco_reader.py
```
This will start the main detection loop with the following defaults:
- Camera: 0
- Resolution: 800x600
- FPS: 120
- Tag size: 0.01 m
- ZMQ port: 9872
- ArUco dictionary: DICT_ARUCO_ORIGINAL

#### Example: Custom Arguments
```bash
python aruco_reader.py --width 1280 --height 960 --fps 60 --tag-size 0.05 --aruco-dict DICT_5X5_100 --port 5555
```

#### Output
The system will detect ArUco markers and publish pose data (x, y, z, roll, pitch, yaw) via ZMQ on the specified port.

## Troubleshooting
- If you see errors about `cv2.aruco`, make sure you have the correct OpenCV package:
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```
- If the camera does not open, check your device index (`--camera`) and permissions.
- For best results, calibrate your camera and use the generated calibration file.

## Citation

# fiducialforcesensor
for ICRA2020 paper: Low-Cost Fiducial-based 6-Axis Force-Torque Sensor

Paper: https://arxiv.org/abs/2005.14250
Supplementary Video and Presentation Slides: https://sites.google.com/view/fiducialforcesensor

If you use this code, please cite the ICRA2020 paper linked above.
