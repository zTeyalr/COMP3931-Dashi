""""
    Preprocessing phase consisting of filtering, resampling, normalization
"""
from scipy.signal import butter, sosfiltfilt, sosfreqz

def butterworth_highpass(ecg_data, cutoff, ecg_frequency, order=3, filter_type='highpass'):
    """"
    Butterworth highpass filter for baseline wander removal
    """
    nyquist_frequency = 0.5 * ecg_frequency #nyquist frequency explanation
    norm_cutoff = cutoff / nyquist_frequency
    sos = butter(order, norm_cutoff, btype=filter_type, analog=False, output='sos')
    filtered_sos = sosfiltfilt(sos, ecg_data, axis=0)

    return filtered_sos

def butterworth_bandpass(ecg_data, cutoff, ecg_frequency, order=3, filter_type='bandpass'):
    """"
    Butterworth bandpass filter
    """
    nyquist_frequency = 0.5 * ecg_frequency #nyquist frequency explanation
    norm_cutoff = [cutoff[0] / nyquist_frequency, cutoff[1] / nyquist_frequency]
    sos = butter(order, norm_cutoff, btype=filter_type, analog=False, output='sos')
    filtered_sos = sosfiltfilt(sos, ecg_data, axis=0)

    return filtered_sos

def resample_signal_frequency():
    """"
    Function to resample data to 500hz
    """
    return 0