import zmq
import numpy as np
import csv
import os
import threading
import time

# ZMQ configuration
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:9872")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Constants
G = 9.81  # Acceleration due to gravity (m/s^2)

# Data collection parameters
data_file = "stiffness_data.csv"
raw_data_file = "stiffness_data_raw.csv"

# Global variables
force = [0, 0, 0]
torque = [0, 0, 0]

# Helper function to calculate force and torque
def calculate_force_torque(mass, length):
    force = mass * G
    torque = force * length
    return force, torque

# Data collection thread
def collect_data(collection_duration):
    data = []
    start_time = time.time()
    
    while time.time() - start_time < collection_duration:
        try:
            pose_data = socket.recv_json(flags=zmq.NOBLOCK)
            data.append(force + torque + [
                pose_data["x"], pose_data["y"], pose_data["z"],
                pose_data["roll"], pose_data["pitch"], pose_data["yaw"]
            ])
        except zmq.Again:
            pass
        
        time.sleep(0.001)  # Small delay to avoid high CPU usage
    
    if data:
        # Calculate the mean of each axis
        mean_data = np.mean(data, axis=0)
        
        # Append the mean data to the CSV file
        with open(data_file, "a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(mean_data)
        
        print("Data collection completed.")
    else:
        print("No data collected.")

# ZMQ data subscription thread
def subscribe_data():
    with open(raw_data_file, "a", newline="") as file:
        csv_writer = csv.writer(file)
        
        while True:
            try:
                pose_data = socket.recv_json(flags=zmq.NOBLOCK)
                timestamp = time.time()
                csv_writer.writerow([timestamp] + force + torque + [
                    pose_data.get("x", np.nan), pose_data.get("y", np.nan), pose_data.get("z", np.nan),
                    pose_data.get("roll", np.nan), pose_data.get("pitch", np.nan), pose_data.get("yaw", np.nan)
                ])
            except zmq.Again:
                pass
            
            time.sleep(0.001)  # Small delay to avoid high CPU usage

# Main CLI loop
def main():
    global force, torque
    
    print("Force/Torque Data Collection CLI")
    print("----------------------------------")
    
    # Create the data files if they don't exist
    if not os.path.exists(data_file):
        with open(data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])
    
    if not os.path.exists(raw_data_file):
        with open(raw_data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Timestamp", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])
    
    # Start the ZMQ data subscription thread
    subscription_thread = threading.Thread(target=subscribe_data)
    subscription_thread.daemon = True
    subscription_thread.start()
    
    while True:
        print("\nOptions:")
        print("1. Enter force and torque values")
        print("2. Enter mass and length (to calculate force and torque)")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            try:
                force = [float(input(f"Enter force in {axis} direction (N): ")) for axis in ["x", "y", "z"]]
                torque = [float(input(f"Enter torque in {axis} direction (N·m): ")) for axis in ["x", "y", "z"]]
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue
        elif choice == "2":
            try:
                mass = float(input("Enter mass (kg): "))
                length = float(input("Enter length (m): "))
                force_val, torque_val = calculate_force_torque(mass, length)
                force = [force_val, 0, 0]
                torque = [0, 0, torque_val]
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        
        try:
            collection_duration = float(input("Enter the data collection duration (in seconds): "))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            continue
        
        input("Press Enter to start data collection...")
        print("Collecting data...")
        
        # Start the data collection thread
        thread = threading.Thread(target=collect_data, args=(collection_duration,))
        thread.start()
        thread.join()  # Wait for the data collection to complete
        
        print("Data collection stopped.")

if __name__ == "__main__":
    main()