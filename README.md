# fiducialforcesensor
for ICRA2020 paper: Low-Cost Fiducial-based 6-Axis Force-Torque Sensor

Paper: https://arxiv.org/abs/2005.14250
Supplementary Video and Presentation Slides: https://sites.google.com/view/fiducialforcesensor

## Installation

1. Install uv (Python package installer):
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

## Usage

1. Run camera calibration (if needed):
```bash
python camera_calibration.py
```

2. Run the ArUco marker detection:
```bash
python aruco_reader.py
```

The system will detect ArUco markers and publish pose data via ZMQ on port 9872.
