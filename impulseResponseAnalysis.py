# %matplotlib widget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, hamming
import easygui as eg
import argparse
from plotly.subplots import make_subplots

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

import plotly.graph_objects as go

# it is all plotting on one. make it each in a separate plot, all 2 rows, 3 cols. the rotation in bottom col corresponding to the axes of its rotation, with labels on each subplots. and also semilog with power in log

def spectral_analysis_plotly(data, channel_names, sampling_rate, nperseg=None):
    """
    Perform spectral analysis using Welch's method and plot the results using Plotly.
    Args:
        data (pandas.DataFrame): DataFrame containing the signals.
        channel_names (list): List of channel names to analyze.
        sampling_rate (float): The sampling rate of the data.
        nperseg (int): Number of points in each segment for Welch's method.

    """
    fig = make_subplots(rows=2, cols=3, subplot_titles=channel_names)
    # fig = go.subplots(rows=2, cols=3, subplot_titles=channel_names)

    for i, channel in enumerate(channel_names):
        # Remove DC offset
        signal_no_dc = remove_dc_offset(data[channel])

        # Compute the Welch's power spectral density estimate
        freqs, psd = welch(signal_no_dc, fs=sampling_rate, window='hamming', nperseg=nperseg, scaling='density')

        # Plot the PSD in a separate subplot
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name=channel), row=row, col=col)

        # # Set the subplot title
        # fig.update_layout(title_text=channel, row=row, col=col)

        # Set the x-axis and y-axis titles
        fig.update_xaxes(title_text='Frequency (Hz)', row=row, col=col)
        fig.update_yaxes(title_text='PSD (Power/Frequency)', row=row, col=col)

        # Set the y-axis to logarithmic scale
        fig.update_yaxes(type='log', row=row, col=col)

    # fig.update_layout(
    #     title='Power Spectral Density',
    #     grid = {
    #         'rows':2,
    #         'columns':3,
    #         'pattern': "independent"
    #     }
    # )

    fig.show()

# def spectral_analysis_plotly(data, channel_names, sampling_rate, nperseg=None):
#     """
#     Perform spectral analysis using Welch's method and plot the results using Plotly.
#     Args:
#         data (pandas.DataFrame): DataFrame containing the signals.
#         channel_names (list): List of channel names to analyze.
#         sampling_rate (float): The sampling rate of the data.
#         nperseg (int): Number of points in each segment for Welch's method.
#     """
#     fig = go.Figure()

#     for i, channel in enumerate(channel_names):
#         # Remove DC offset
#         signal_no_dc = remove_dc_offset(data[channel])

#         # Compute the Welch's power spectral density estimate
#         freqs, psd = welch(signal_no_dc, fs=sampling_rate, window='hamming', nperseg=nperseg, scaling='density')

#         # Plot the PSD
#         fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name=channel))

#     fig.update_layout(
#         title='Power Spectral Density',
#         xaxis_title='Frequency (Hz)',
#         yaxis_title='PSD (Power/Frequency)',
#         grid = {
#             'rows':2,
#             'columns':3,
#             'pattern': "independent"
#         }
#     )

#     fig.show()



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
    spectral_analysis_plotly(data, args.channels, args.sampling_rate, nperseg=nperseg_welch)
    # spectral_analysis_welch(data, args.channels, args.sampling_rate, nperseg=nperseg_welch)


if __name__ == '__main__':
    main()

