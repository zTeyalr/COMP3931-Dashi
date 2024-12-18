""""
    Preprocessing phase consisting of filtering, resampling, normalization
"""

from scipy.signal import butter, sosfiltfilt, decimate, resample_poly
from torch.utils.data import random_split, Subset
import numpy as np
from data_loader import get_frequency, get_adc_gains, get_leads_num, get_lead_names, get_baseline, load_ecg_data, extract_features, ECGDataset
from statistics import mean
import random
import os
import gc
import matplotlib.pyplot as plt


LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def zero_padding(ecg_data, standard_length = 8192):
    """"
    Zero padding to make even signals, choosing 2^13 samples
    Random Truncating for signals with over 2^13 samples
    """
    if ecg_data.shape[1] < standard_length:
        pad_len = standard_length - ecg_data.shape[1]
        padding = np.zeros((ecg_data.shape[0], pad_len))
        ecg_data = np.concatenate([ecg_data, padding], axis=1) # Adds the zeros at the end of the signal if needed

    elif ecg_data.shape[1] > standard_length:
        index = np.random.randint(0, ecg_data.shape[1] - standard_length) # Random select a portion of the signal
        ecg_data = ecg_data[:, index:index + standard_length]
    
    if ecg_data.shape[0] < 12:
        pad_len = 12 - ecg_data.shape[0]
        padding = np.zeros((pad_len, ecg_data.shape[1]))
        ecg_data = np.concatenate([ecg_data, padding], axis=0)

    return ecg_data

def masked_data(ecg_data):
    num_non_zero = np.count_nonzero(ecg_data, axis=1)
    mask = np.zeros((ecg_data.shape[0], len(LEADS)))

    for i, sample_length in enumerate(num_non_zero):
        mask[i, : sample_length] = 1
    return mask

def mask_data(header_file):
    present_leads = get_lead_names(header_file)
    mask = []
    for lead in LEADS:
        if lead in present_leads:
            mask.append(1)
        else:
            mask.append(0)

    return mask

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
        return ecg_data
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

    return norm_data


def preprocess_data(ecg_data, ecg_frequency, adc_gains, baseline, mean=None, std=None, target_frequency=500):
    """
        Apply the preprocessing phase to the data
    """

    adc_gains = adc_gains[:, np.newaxis]
    baseline = baseline[:, np.newaxis]

    ecg_data_c = (ecg_data - baseline) / adc_gains

    ecg_preprocessed = butterworth_highpass(ecg_data_c, cutoff=8, ecg_frequency=ecg_frequency)
    ecg_preprocessed = butterworth_bandpass(ecg_preprocessed, [0.2, 37], ecg_frequency)
    ecg_preprocessed = resample_signal_frequency(ecg_preprocessed, ecg_frequency, target_frequency)
    ecg_preprocessed = z_normalize(ecg_preprocessed)
    ecg_preprocessed = zero_padding(ecg_preprocessed)

    return ecg_preprocessed



def preprocess_and_save(ecg_dir, output_dir):

    ecg_data, headers, ecg_data_base_filenames, age_array = load_ecg_data(ecg_dir)
    dataset = ECGDataset(ecg_data_all=ecg_data, ecg_data_paths=ecg_data_base_filenames ,header_all=headers, transform=None)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size =int(0.8 * len(indices)) # Training
    val_size = int(0.1 * len(indices)) # Validation
    test_size = len(indices) - train_size - val_size # Testing

    train_indices, val_indices, test_indices = random_split(indices, [train_size, val_size, test_size])

    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, train_indices)

    train_headers = [headers[i] for i in train_indices]
    val_headers = [headers[i] for i in val_indices]
    test_headers = [headers[i] for i in test_indices]
    
    train_ecg_data_paths = [ecg_data_base_filenames[i] for i in train_indices]
    val_ecg_data_paths = [ecg_data_base_filenames[i] for i in val_indices]
    test_ecg_data_paths = [ecg_data_base_filenames[i] for i in test_indices]

    train_mean_age = mean(age_array[i] for i in train_indices)
    val_mean_age = mean(age_array[i] for i in val_indices)
    test_mean_age = mean(age_array[i] for i in test_indices)

    del ecg_data
    del headers
    del ecg_data_base_filenames
    del age_array
    del dataset
    gc.collect()

    batch_size = 128
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    features_dir = os.path.join(output_dir, "ecg_features")
    train_mean = 0
    train_std = 0
    num_samples = 0
    target_frequency = 500
    subset_name = ['train', 'val', 'test']

    # TRAIN DATASET
    for i in range(0, len(train_indices), batch_size):
        
        end_idx = min(i + batch_size, len(train_indices)) #Handle last batch size

        batch_indices = train_indices.indices[i: end_idx]
        batch_data = [train_data[j] for j in range(len(batch_indices))]
        batch_headers = [train_headers[j] for j in range(len(batch_indices))]

        batch_preprocessed = [preprocess_data(batch_data[k][0], get_frequency(batch_headers[k]), get_adc_gains(batch_headers[k],
                                                get_leads_num(batch_headers[k])), get_baseline(batch_headers[k])) 
                                            for k in range(len(batch_indices))]
        batch_features = [extract_features(data, header, target_frequency, train_mean_age) for data, header in zip(batch_preprocessed, batch_headers)]

        batch_mean = np.mean(batch_preprocessed)
        batch_std = np.std(batch_preprocessed)
        batch_size = len(batch_preprocessed)

        # Accumulate the mean and variance
        train_mean = (train_mean * num_samples + batch_mean * batch_size) / (num_samples + batch_size)
        train_std = np.sqrt((train_std**2 * num_samples + batch_std**2 * batch_size) / (num_samples + batch_size))
        num_samples += batch_size
        
        save_preprocessed_batch(batch_preprocessed, batch_features, train_dir, features_dir, train_ecg_data_paths, subset_name[0], i)
        del batch_data
        del batch_headers
        del batch_preprocessed
        del batch_features
    print("DONE TRAINING")

    #VALIDATION DATASET
    for i in range(0, len(val_indices), batch_size):

        end_idx = min(i + batch_size, len(val_indices)) #Handle last batch size

        batch_indices = val_indices.indices[i: end_idx]
        batch_data = [val_data[j] for j in range(len(batch_indices))]
        batch_headers = [val_headers[j] for j in range(len(batch_indices))]

        batch_preprocessed = [preprocess_data(batch_data[k][0], get_frequency(batch_headers[k]), get_adc_gains(batch_headers[k],
                                                get_leads_num(batch_headers[k])), get_baseline(batch_headers[k]), train_mean, train_std)
                                            for k in range(len(batch_indices))]
        batch_features = [extract_features(data, header, target_frequency, val_mean_age) for data, header in zip(batch_preprocessed, batch_headers)]

        save_preprocessed_batch(batch_preprocessed, batch_features, val_dir, features_dir, val_ecg_data_paths, subset_name[1], i)

        del batch_data
        del batch_headers
        del batch_preprocessed
        del batch_features
    print("DONE VAL")

    #TEST DATASET
    for i in range(0, len(test_indices), batch_size):

        end_idx = min(i + batch_size, len(test_indices)) #Handle last batch size

        batch_indices = test_indices.indices[i: end_idx]
        batch_data = [test_data[j] for j in range(len(batch_indices))]
        batch_headers = [test_headers[j] for j in range(len(batch_indices))]

        batch_preprocessed = [preprocess_data(batch_data[k][0], get_frequency(batch_headers[k]), get_adc_gains(batch_headers[k],
                                                get_leads_num(batch_headers[k])), get_baseline(batch_headers[k]), train_mean, train_std) 
                                            for k in range(len(batch_indices))]
        batch_features = [extract_features(data, header, target_frequency, test_mean_age) for data, header in zip(batch_preprocessed, batch_headers)]
        
        save_preprocessed_batch(batch_preprocessed, batch_features, test_dir, features_dir, test_ecg_data_paths, subset_name[2], i)

        del batch_data
        del batch_headers
        del batch_preprocessed
        del batch_features
    print("DONE TEST")

def save_preprocessed_batch(preprocessed_batch, batch_features, output_dir, feature_dir, filename, subset_name, start_idx):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    for k, (preprocessed_data, features) in enumerate(zip(preprocessed_batch, batch_features)):
        output_filename = os.path.join(output_dir, f"{filename[start_idx + k]}.npy")
        np.save(output_filename, preprocessed_data)

        features_filename = os.path.join(feature_dir, f"{filename[start_idx + k]}_{subset_name}_features.npz")
        np.savez(features_filename, **features)
