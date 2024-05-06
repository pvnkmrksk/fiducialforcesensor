# %matplotlib widget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, hamming
import easygui as eg
import argparse

def load_data(filepath):
    """
    Load data from a CSV file.
    Args:
        filepath (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)

def remove_dc_offset(signal):
    """
    Remove the DC offset from the signal.
    Args:
        signal (np.ndarray): The input signal.
    Returns:
        np.ndarray: Signal with the DC offset removed.
    """
    return signal - np.mean(signal)

def spectral_analysis_welch(data, channel_names, sampling_rate, nperseg=None):
    """
    Perform spectral analysis using Welch's method.
    Args:
        data (pandas.DataFrame): DataFrame containing the signals.
        channel_names (list): List of channel names to analyze.
        sampling_rate (float): The sampling rate of the data.
        nperseg (int): Number of points in each segment for Welch's method.
    """
    plt.figure(figsize=(15, 12))
    
    for i, channel in enumerate(channel_names):
        # Remove DC offset
        signal_no_dc = remove_dc_offset(data[channel])
        
        # Compute the Welch's power spectral density estimate
        freqs, psd = welch(signal_no_dc, fs=sampling_rate, window='hamming', nperseg=nperseg, scaling='density')
        
        #normalize the PSD to area = 1
        # psd = psd / np.sum(psd)

        # Plot the PSD
        plt.subplot(len(channel_names), 1, i + 1)
        # plt.plot(freqs, psd)
        plt.semilogy(freqs, psd)

        plt.title(f'Power Spectral Density of {channel}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (Power/Frequency)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('spectral_analysis_welch.png')
    plt.show()



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Perform spectral analysis on a CSV file.')

    # Add the command line arguments
    parser.add_argument('-f','--filepath', type=str, help='Path to the CSV file', default='~/impulse4.csv')
    parser.add_argument('-s', '--sampling_rate', type=float, help='Sampling rate of the data. If not provided, it will be estimated.')
    parser.add_argument('-c', '--channels', nargs='+', default=['/x', '/y', '/z', '/pitch', '/yaw', '/roll'], help='List of channel names to analyze')

    # Parse the command line arguments
    args = parser.parse_args()

    # Load the data
    data = load_data(args.filepath)

    # Estimate the sampling rate if not provided
    if args.sampling_rate is None:
        try:
            args.sampling_rate = 1 / np.mean(np.diff(data['__time']))
        except:
            print("Error: Could not estimate the sampling rate. Setting it to 100 Hz.")
            args.sampling_rate = 100

    # Specify the number of points used in each segment for Welch's method
    nperseg_welch = len(data) // 8

    # Perform spectral analysis using Welch's method
    spectral_analysis_welch(data, args.channels, args.sampling_rate, nperseg=nperseg_welch)

if __name__ == '__main__':
    main()

