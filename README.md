## Introduction
# Fiducial Force Sensor

This project implements a low-cost, vision-based 6-axis force-torque sensor using ArUco fiducial markers and a standard camera. The approach tracks the 3D pose of a marker attached to a compliant structure, inferring forces and torques from observed displacements. The system leverages OpenCV's ArUco module for robust marker detection and pose estimation. Calibration routines are provided for both the camera and the flexure structure. Real-time pose data is published via ZMQ for integration with robotics and data acquisition systems.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pvnkmrksk/fiducialforcesensor.git
   cd fiducialforcesensor
   ```

2. Install [uv](https://github.com/astral-sh/uv) (Python package installer):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. Install dependencies:
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

#### Available Command Line Arguments
   ```bash
   python aruco_reader.py --help
   ```
   ```
   usage: aruco_reader.py [-h] [--tag-size TAG_SIZE] [--camera CAMERA] [--width WIDTH]
                          [--height HEIGHT] [--fps FPS] [--exposure EXPOSURE] [--gain GAIN]
                          [--gamma GAMMA] [--brightness BRIGHTNESS] [--contrast CONTRAST]
                          [--aruco-dict {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,
                          DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,
                          DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,
                          DICT_7X7_250,DICT_7X7_1000,DICT_ARUCO_ORIGINAL}]
                          [--subset-id SUBSET_ID] [--baseline-frames BASELINE_FRAMES]
                          [--port PORT] [--min-pixel-size MIN_PIXEL_SIZE]
                          [--max-pixel-size MAX_PIXEL_SIZE] [--debug]

   ArUco marker detection and pose estimation

   options:
     -h, --help            show this help message and exit
     --tag-size TAG_SIZE   Size of the ArUco tag in meters (default: 0.01)
     --camera CAMERA       Camera device ID (default: 0)
     --width WIDTH         Camera width resolution (default: 800)
     --height HEIGHT       Camera height resolution (default: 600)
     --fps FPS             Camera FPS (default: 120)
     --exposure EXPOSURE   Camera exposure (default: 1)
     --gain GAIN           Camera gain (default: 1)
     --gamma GAMMA         Camera gamma (default: 72)
     --brightness BRIGHTNESS
                           Camera brightness (default: 0)
     --contrast CONTRAST   Camera contrast (default: 32)
     --aruco-dict {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,
                   DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,
                   DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,
                   DICT_7X7_1000,DICT_ARUCO_ORIGINAL}
                           ArUco dictionary to use (default: DICT_ARUCO_ORIGINAL)
     --subset-id SUBSET_ID
                           ArUco marker subset ID for DICT_ARUCO_ORIGINAL (default: None)
     --baseline-frames BASELINE_FRAMES
                           Number of frames to use for baseline calculation (default: 500)
     --port PORT           ZMQ port for publishing pose data (default: 9872)
     --min-pixel-size MIN_PIXEL_SIZE
                           Minimum marker side length in pixels (default: 250)
     --max-pixel-size MAX_PIXEL_SIZE
                           Maximum marker side length in pixels (default: 700)
     --debug               Enable debug visualization of rejected markers

   Examples:
     python aruco_reader.py --tag-size 0.05 --camera 1 --width 640 --height 480
     python aruco_reader.py --aruco-dict DICT_5X5_100 --port 5555
   ```

#### Example: Custom Arguments
   ```bash
   python aruco_reader.py --width 1280 --height 960 --fps 60 --tag-size 0.05 --aruco-dict DICT_5X5_100 --port 5555
   ```

#### Output
The system will detect ArUco markers and publish pose data (x, y, z, roll, pitch, yaw) via ZMQ on the specified port.

## Visualization with PlotJuggler

For real-time visualization of the pose data, we recommend using [PlotJuggler](https://github.com/facontidavide/PlotJuggler), a powerful time series visualization tool.

### Install PlotJuggler

**Ubuntu (with ROS support):**
```bash
sudo snap install plotjuggler
```

**Windows:**
Download the installer from [PlotJuggler releases](https://github.com/facontidavide/PlotJuggler/releases)

**From source:**
```bash
git clone https://github.com/facontidavide/PlotJuggler.git
cd PlotJuggler
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Connect to ZMQ Data Stream

1. Launch PlotJuggler
2. Go to **Data Sources** → **Add DataStreamer** → **ZeroMQ**
3. Configure the connection:
   - **URL**: `tcp://localhost:9872` (or your custom port)
   - **Topic**: Leave empty (receives all topics)
4. The pose data (x, y, z, roll, pitch, yaw) will appear as time series plots

PlotJuggler provides advanced features like:
- Real-time plotting with thousands of data points
- Data transformation and filtering
- Layout saving and sharing
- Export capabilities

## Hardware Design

The flexure system design is available as an Onshape CAD model:

**[Flexure System CAD Model](https://cad.onshape.com/documents/3e860df39a3a0136b6650a0b/w/78ad68392459b8d68de64f2d/e/6d504a39dfdeb2eaddb9b7d0?configuration=Dual_Marker%3Dfalse%3BDual_Marker_Size%3D0.004%2Bmeter%3BFit_tolerance%3D0.1%3BFlexure_Height%3D0.035%2Bmeter%3BFlexure_Loop_Diameter%3D0.01%2Bmeter%3BFlexure_Loop_Fillet%3D0.001%2Bmeter%3BFlexure_Thickness_Height%3D0.0014%2Bmeter%3BFlexure_Thickness_Width%3D0.0014%2Bmeter%3BFlexure_Width%3D0.035%2Bmeter%3BPlatform_Size%3D0.015%2Bmeter%3BPlatform_Wall_Thickness%3D8.0E-4%2Bmeter%3BVersion%3D&renderMode=0&uiState=6852e49db5b2c71c25c7c021)**

The design includes:
- Configurable flexure parameters (height, width, thickness)
- Platform size and wall thickness options
- Dual marker support (configurable)
- Fit tolerance settings
- 3D printable geometry

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
