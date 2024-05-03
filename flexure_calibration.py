import datetime
import json
import zmq
import numpy as np
import csv
import os
import threading
import time

# ZMQ configuration
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:9873")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Constants
G = 9.81  # Acceleration due to gravity (m/s^2)

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
    if mass == 0:
        return 0
    return mass * G / 1000

def calculate_torque(mass, length):
    """
    Calculate the torque based on the given mass and length.

    Args:
        mass (float): Mass in grams (g).
        length (float): Length in millimeters (mm).

    Returns:
        float: Torque in Newton-meters (N·m).
    """
    if mass == 0 or length == 0:
        return 0
    return calculate_force(mass) * length / 1000

def collect_data(collection_duration):
    """
    Collect data for the specified duration.

    Args:
        collection_duration (float): Duration of data collection in seconds.
    """
    data = []
    start_time = time.time()

    with open(raw_data_file, "a", newline="") as file:
        csv_writer = csv.writer(file)

        while time.time() - start_time < collection_duration:
            try:
                pose_data = socket.recv_json(flags=zmq.NOBLOCK)
                timestamp = time.time()
                row = [timestamp] + force + torque + [
                    pose_data.get("x", np.nan), pose_data.get("y", np.nan), pose_data.get("z", np.nan),
                    pose_data.get("roll", np.nan), pose_data.get("pitch", np.nan), pose_data.get("yaw", np.nan)
                ]
                csv_writer.writerow(row)
                data.append(row[1:])  # Exclude the timestamp from the data for averaging
            except zmq.Again:
                pass
            except Exception as e:
                print(f"Error during data collection: {e}")

            time.sleep(0.001)

    if data:
        print("Data being averaged.")

        try:
            data_array = np.array(data)
            mean_data = np.nanmean(data_array, axis=0)
            with open(data_file, "a", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(mean_data)
        except Exception as e:
            print(f"Error during data averaging: {e}")

        print("Data collection completed.")
    else:
        print("No data collected.")

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
        
        # convert rotation to radians
        A_disp[:, 3:] = np.radians(A_disp[:, 3:])

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
    global force, torque, data_file, raw_data_file
    
    print("Force/Torque Data Collection CLI")
    print("----------------------------------")

    # ask user if they want a new file or to append to an existing file
    file_option = input("Do you want to create a new file or append to an existing file? (Enter 'new' or 'append'): ")
    
    #new or n 
    if file_option in ["new", "n"] or not os.path.exists(data_file) or not os.path.exists(raw_data_file):
        # Create new data files
        #append time stamp to the file name
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_file = f"stiffness_data_{now}.csv"
        raw_data_file = f"stiffness_data_raw_{now}.csv"

        with open(data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])
        
        with open(raw_data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Timestamp", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])

        print(f"New data files created: {data_file}, {raw_data_file}")

    elif file_option in ["append", "a"]:
        # Check if the data files already exist
        if not os.path.exists(data_file) or not os.path.exists(raw_data_file):
            print("Data files not found. Creating new files instead.")
            
            with open(data_file, "w", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])
            
            with open(raw_data_file, "w", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Timestamp", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])

        print(f"Appending data to existing files: {data_file}, {raw_data_file}")
    else:
        print("Invalid option. Creating new files instead.")
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_file = f"stiffness_data_{now}.csv"
        raw_data_file = f"stiffness_data_raw_{now}.csv"

        with open(data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])
        
        with open(raw_data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Timestamp", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])

        print(f"New data files created: {data_file}, {raw_data_file}")
    

    
    
    # Create the data files if they don't exist
    if not os.path.exists(data_file):
        with open(data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])
    
    if not os.path.exists(raw_data_file):
        with open(raw_data_file, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Timestamp", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "roll", "pitch", "yaw"])
    
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
                force = []
                torque = []
                for axis in ["x", "y", "z"]:
                    mass_force = float(input(f"Enter mass for {axis}-axis force (g): "))
                    force.append(calculate_force(mass_force))

                    mass_torque = float(input(f"Enter mass for {axis}-axis torque (g): "))
                    length_torque = float(input(f"Enter length for {axis}-axis torque (mm): "))
                    torque.append(calculate_torque(mass_torque, length_torque))
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
        
        # Start the data collection
        collect_data(collection_duration)
        
        print("Data collection stopped.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce/Torque Data Collection CLI terminated.")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        socket.close()
        context.term()
        exit(0)