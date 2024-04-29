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

def calculate_force(mass):
    """
    Calculate the force based on the given mass.

    Args:
        mass (float): Mass in grams (g).

    Returns:
        float: Force in Newtons (N).
    """
    return mass * G / 1000  # Convert grams to kilograms

def calculate_torque(mass, length):
    """
    Calculate the torque based on the given mass and length.

    Args:
        mass (float): Mass in grams (g).
        length (float): Length in millimeters (mm).

    Returns:
        float: Torque in Newton-meters (N·m).
    """
    return calculate_force(mass) * length / 1000  # Convert millimeters to meters

def collect_data(collection_duration):
    """
    Collect data for the specified duration.

    Args:
        collection_duration (float): Duration of data collection in seconds.
    """
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

def subscribe_data():
    """
    Subscribe to the ZMQ data stream and save the raw data to a CSV file.
    """
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

def perform_analysis():
    """
    Perform stiffness matrix analysis based on the collected data.
    """
    try:
        # Load the collected data from the CSV file
        data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
        
        if len(data) == 0:
            print("No data available for analysis. Please collect data first.")
            return
        
        # Split the data into force/torque and displacement/rotation matrices
        A_force = data[:, :6]
        A_disp = data[:, 6:]
        
        # Estimate the stiffness matrix using least-squares
        k = np.linalg.lstsq(A_force, A_disp, rcond=None)[0]
        
        # Print the estimated stiffness matrix
        print("\nEstimated Stiffness Matrix (K):")
        print(k)
        
        # Save the stiffness matrix to a file
        np.savetxt("stiffness_matrix.txt", k, delimiter=',')
        print("Stiffness matrix saved to 'stiffness_matrix.txt'.")
    except FileNotFoundError:
        print("Data file not found. Please collect data first.")
    except ValueError as ve:
        print(f"Error during analysis: {ve}")

def main():
    """
    Main function to run the Force/Torque Data Collection CLI.
    """
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
        print("1. Enter force and torque values (in N and N·m)")
        print("2. Enter mass values (in grams) for force and torque")
        print("3. Perform stiffness matrix analysis")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            try:
                force = [float(input(f"Enter force in {axis} direction (N): ")) for axis in ["x", "y", "z"]]
                torque = [float(input(f"Enter torque in {axis} direction (N·m): ")) for axis in ["x", "y", "z"]]
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue
        elif choice == "2":
            try:
                force = [calculate_force(float(input(f"Enter mass for {axis}-axis force (g): "))) for axis in ["x", "y", "z"]]
                torque = [calculate_torque(
                    float(input(f"Enter mass for {axis}-axis torque (g): ")),
                    float(input(f"Enter length for {axis}-axis torque (mm): "))
                ) for axis in ["x", "y", "z"]]
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue
        elif choice == "3":
            perform_analysis()
            continue
        elif choice == "4":
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