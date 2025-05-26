#!/usr/bin/env python3
"""
Example script demonstrating dual marker ArUco detection.

This script shows how to use the dual marker functionality for more robust
pose estimation using two markers positioned at 90 degrees to each other.

The markers should be adjacent and perpendicular - the separation distance
is automatically calculated from the tag size.

Usage examples:
    # Basic dual marker mode (auto-detect markers)
    python dual_marker_example.py
    
    # Dual marker mode with specific marker IDs
    python dual_marker_example.py --marker-id-1 64 --marker-id-2 65
    
    # Dual marker mode with custom tag size
    python dual_marker_example.py --marker-id-1 64 --marker-id-2 65 --tag-size 0.02
"""

import subprocess
import sys

def run_dual_marker_detection():
    """Run the ArUco reader in dual marker mode with example parameters."""
    
    # Example 1: Basic dual marker mode with auto-detection
    print("=== Example 1: Basic Dual Marker Mode ===")
    print("Running dual marker detection with auto-detection...")
    print("Command: python aruco_reader.py --dual-marker --debug")
    print()
    
    # Example 2: Dual marker mode with specific IDs
    print("=== Example 2: Dual Marker with Specific IDs ===")
    print("Running dual marker detection with specific marker IDs...")
    print("Command: python aruco_reader.py --dual-marker --marker-id-1 64 --marker-id-2 65 --debug")
    print()
    
    # Example 3: Dual marker mode with custom parameters
    print("=== Example 3: Dual Marker with Custom Tag Size ===")
    print("Running dual marker detection with custom tag size...")
    print("Command: python aruco_reader.py --dual-marker --marker-id-1 64 --marker-id-2 65 --tag-size 0.02 --debug")
    print()
    
    # Ask user which example to run
    choice = input("Which example would you like to run? (1/2/3): ").strip()
    
    if choice == "1":
        cmd = [
            "python", "aruco_reader.py",
            "--dual-marker",
            "--debug",
            "--tag-size", "0.01",
            "--baseline-frames", "100"
        ]
    elif choice == "2":
        cmd = [
            "python", "aruco_reader.py",
            "--dual-marker",
            "--marker-id-1", "64",
            "--marker-id-2", "65",
            "--debug",
            "--tag-size", "0.01",
            "--baseline-frames", "100"
        ]
    elif choice == "3":
        cmd = [
            "python", "aruco_reader.py",
            "--dual-marker",
            "--marker-id-1", "64",
            "--marker-id-2", "65",
            "--tag-size", "0.02",  # Larger tag size example
            "--debug",
            "--baseline-frames", "100"
        ]
    else:
        print("Invalid choice. Exiting.")
        return
    
    print(f"Running command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop the detection.")
    print("Press ESC in the camera window to exit.")
    print()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
    except FileNotFoundError:
        print("Error: aruco_reader.py not found. Make sure you're in the correct directory.")
    except Exception as e:
        print(f"Error running detection: {e}")

def print_dual_marker_info():
    """Print information about dual marker setup."""
    print("=== Dual Marker Setup Information ===")
    print()
    print("Dual marker mode uses two ArUco markers positioned at 90 degrees")
    print("to each other for more robust pose estimation, particularly for")
    print("pitch and roll measurements.")
    print()
    print("Key benefits:")
    print("- More robust pitch and roll estimation")
    print("- Reduced sensitivity to marker occlusion")
    print("- Better performance when one marker is at an angle")
    print()
    print("Setup requirements:")
    print("- Two ArUco markers from the same dictionary")
    print("- Markers positioned at 90° to each other (adjacent and perpendicular)")
    print("- Both markers should be visible to the camera")
    print("- Accurate tag size measurement (separation calculated automatically)")
    print()
    print("Command line options:")
    print("  --dual-marker              Enable dual marker mode")
    print("  --marker-id-1 ID           Specific ID for first marker (optional)")
    print("  --marker-id-2 ID           Specific ID for second marker (recommended)")
    print("  --tag-size SIZE            Size of markers in meters (affects calculated separation)")
    print("  --debug                    Show debug visualization")
    print()

if __name__ == "__main__":
    print("ArUco Dual Marker Detection Example")
    print("===================================")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        print_dual_marker_info()
    else:
        print_dual_marker_info()
        print()
        run_dual_marker_detection() 