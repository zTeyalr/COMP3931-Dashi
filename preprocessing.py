""""
    Preprocessing phase consisting of filtering, resampling, normalization
"""
from scipy.signal import butter, sosfiltfilt, decimate, resample_poly
import numpy as np

def zero_padding(ecg_data, standard_length = 8192):
    """"
    Zero padding to make even signals, choosing 2^13 samples
    """
    if ecg_data.shape[1] < standard_length:
        pad_len = standard_length - ecg_data.shape[1]
        padding = np.zeros((ecg_data.shape[0], pad_len))
        pad_data = np.concatenate([ecg_data, padding], axis=1)

        return pad_data

def butterworth_highpass(ecg_data, cutoff, ecg_frequency, order=3):
    """"
    Butterworth highpass filter for baseline wander removal
    """
    nyquist_frequency = 0.5 * ecg_frequency # nyquist frequency to prevent aliasing
    norm_cutoff = cutoff / nyquist_frequency
    sos = butter(order, norm_cutoff, btype='highpass', analog=False, output='sos')
    filtered_sos = sosfiltfilt(sos, ecg_data, axis=1)

    return filtered_sos

def butterworth_bandpass(ecg_data, cutoff, ecg_frequency, order=3):
    """"
    Butterworth bandpass filter to remove noise and artifacts
    """
    nyquist_frequency = 0.5 * ecg_frequency # nyquist frequency to prevent aliasing
    norm_cutoff = [cutoff[0] / nyquist_frequency, cutoff[1] / nyquist_frequency]
    sos = butter(order, norm_cutoff, btype='bandpass', analog=False, output='sos')
    filtered_sos = sosfiltfilt(sos, ecg_data, axis=1)

    return filtered_sos

def resample_signal_frequency(ecg_data, ecg_frequency, target_frequency):
    """"
    Function to resample data to 500hz
    """
    if ecg_frequency == target_frequency: # no need to resample the signal
        return 0
    if ecg_frequency > target_frequency: # decimation phase for frequencies higher than 500
        downsampling_factor = ecg_frequency // target_frequency
        resampled_data = decimate(ecg_data, downsampling_factor, ftype='fir', axis=0, zero_phase=True)
    else:
        up = target_frequency // ecg_frequency
        resampled_data = resample_poly(ecg_data, up, down=1, axis=0) #upsampling
    return resampled_data

def z_normalize(ecg_data):
    """"
    Normalize data using z-score
    """
    # Calculate the mean of data
    # keepdims set to True to maintain array dimension
    
    # Normalizing seems to clip too much NEED REVISION AFTER MODEL
    mean = np.mean(ecg_data, axis=1, keepdims=True)
    std = np.std(ecg_data, axis=1, keepdims=True)

    std[std == 0] = 1e-10 # Prevent division by zero Error
    norm_data = (ecg_data - mean) / std

    outliers = np.abs(norm_data) > 3 # Check outliers above a common threshold for now

    return norm_data, outliers
