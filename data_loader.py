""" LOAD DATA AND RETRIEVE INFORMATION
    Some of these function have been taken from the official
    Physionet Challenge 2021 Github helper_code.py template"""

import os
import scipy.io as sc
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from ecgdetectors import Detectors
from hrvanalysis import get_time_domain_features, remove_outliers, interpolate_nan_values
import matplotlib.pyplot as plt


class ECGDataset(Dataset):
    def __init__(self, ecg_data, headers, features, labels):
        self.ecg_data = ecg_data
        self.headers = headers
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        try:
            signal = self.ecg_data[idx]
            header = self.headers[idx]
            features = self.features[idx]
            
            #print("SIGNAL DATASET ECG", header, signal)

            snomet_code = get_patient_data(header)['Dx']
            scored_labels, unscored_labels = get_snomed_list()
            combined_labels = np.concatenate((scored_labels, unscored_labels), axis = 0)
            
            #all_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            labels = np.zeros(len(combined_labels))

            if ',' in snomet_code:
                codes = snomet_code.split(',')
                for code in codes:
                    idx = np.where(combined_labels == np.int64(code))[0]
                    labels[idx] = 1
            else:
                idx = np.where(combined_labels == np.int64(snomet_code))[0]
                labels[idx] = 1

            #CONVERT TO TENSOR
            signal = torch.from_numpy(signal).float()
            labels = torch.tensor(labels).float()
            features = torch.tensor(features).float()
            
            return signal, labels, features
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return None, None, None


class FocalLoss(nn.Module):
    def __init__(self, gamma=4, alpha=None, reduction='mean', scored_indices=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.scored_indices = scored_indices

    def forward(self, inputs, targets):
        # Standard BCEWithLogitsLoss (with optional class weighting)
        if self.scored_indices is not None:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=self.weights, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Focal Loss modulation
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss if self.alpha is not None else (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Load .mat files
def load_mat(mat_file):
    """Function loading mat files"""
    mat_data = sc.loadmat(mat_file)
    ecg_signals = mat_data['val']

    return {'ecg_signal': ecg_signals}


def load_pre_processed_mat(mat_file):
    mat_data = np.load(mat_file)
    return mat_data



def load_ecg_data(data_dir, preprocessed = False):
    """
    Loads ECG data, headers, and patient ages from the specified directory.
    """
    ecg_data = []
    headers = []
    ecg_data_base_filenames = []
    age_array = []
    file_extension = '.mat.npy' if preprocessed else '.mat'
    
    for root, _, filename in os.walk(data_dir):
        for file in filename:
            if file.endswith(file_extension):
                mat_file = os.path.join(root, file)
                ecg_data_base_filenames.append(file.split('.')[0])
                

                if preprocessed:
                    ecg_data_loaded = load_pre_processed_mat(mat_file)
                else:
                    ecg_data_loaded = load_mat(mat_file)
                    hea_file = mat_file.replace(file_extension, '.hea')
                    header = load_header(hea_file)
                    headers.append(header)
                    ecg_data.append(ecg_data_loaded['ecg_signal'])
    if len(headers) > 0:
        for header in headers:
            try:
                age = int(get_patient_data(header)['Age'])
                age_array.append(age)
            except ValueError:
                age_array.append(np.nan)
            
    return ecg_data, headers, ecg_data_base_filenames, age_array



# Load .hea files
def load_header(header_file):
    """Function loading header files"""
    with open(header_file, 'r', encoding="utf-8") as f:
        header = f.read()
    return header

# Get info from header file as wfdb.rdsamp does not work with custom headers

def load_features_folder(features_folder, subset_name):
    features_list = []
    for feature_file in os.listdir(features_folder):
        if feature_file.endswith('.npz'):
            features_path = os.path.join(features_folder, feature_file)

            with np.load(features_path, allow_pickle=True) as data:
                features = dict(data)
            features_list.append(features)
    return features_list


def load_single_feature(features_path):
    features_list = []
    with np.load(features_path, allow_pickle=True) as data:
        features = dict(data)
        features_list.append(features)
    return features_list


def get_snomed_list():
    
    scored_labels = pd.read_csv('dx_mapping_scored.csv')
    
    unscored_labels = pd.read_csv('dx_mapping_unscored.csv')
    scored_dx = np.array(scored_labels['SNOMEDCTCode'].to_list(), dtype=np.int64)
    unscored_dx = np.array(unscored_labels['SNOMEDCTCode'].to_list(), dtype=np.int64)
    
    return scored_dx, unscored_dx



def get_line(header_file, line_num):
    """Function to get a specific line in the file"""
    lines = header_file.splitlines()
    try:
        line = lines[line_num]
        return line
    except:
        print(f"Could not find line")


def get_lines(header_file):
    lines = header_file.splitlines()
    return lines


# Get Record ID
def get_id(header_file):
    header_id = None
    
    try:
        header_id = get_line(header_file, 0).split(' ')[0]
        return header_id
    except IndexError:
        print(f"Out of bound Index")
        
        
def get_leads_num(header_file):
    leads_num = None
    
    try:
        leads_num = int(get_line(header_file, 0).split(' ')[1])
        return leads_num
    except IndexError:
        print(f"Out of bound Index")
       
        
def get_frequency(header_file):
    frequency = None
    
    try:
        frequency = get_line(header_file, 0).split(' ')[2]
        return int(frequency)

    except IndexError:
        print(f"Out of bound Index")


def get_sample_num(header_file):
    sample_num = None
    
    try:
        sample_num = get_line(header_file, 0).split('')[3]
        return sample_num
    except IndexError:
        print(f"Out of bound Index")
        

def get_patient_data(header_file):
    lines = get_lines(header_file)
    
    patient_data = {} # Initialize dictionary that will contain patient information

    for line in lines:
        if line.startswith("#"):
            # Split line to get key and value
            x = line.split(":")
            try:   
                key = x[0].lstrip("#").strip()
                try:
                    value = x[1].strip()
                except:
                    value = None 
                patient_data[key] = value
            except IndexError:
                print(f"Out of bound Index")
            except ValueError:
                print(f"Error parsing")

    return patient_data

def get_lead_names(header_file):
    lines = get_lines(header_file)[1:]
    lead_names = []
    num_leads = int(get_leads_num(header_file))
    counter = 0
    for line in lines:
        line = line.split()
        counter += 1
        if counter > num_leads:
            break
        lead_names.append(line[-1])
    return lead_names


def get_adc_gains(header_file, leads):
    adc_gains = np.zeros(leads)
    lines = get_lines(header_file)[1:]
    counter = 0
    for line in lines:
        line = line.split()
        adc_gains[counter] = float(line[2].split('/')[0])
        counter += 1
        if counter == leads:
            break
        
    return adc_gains

def get_baseline(header_file):
    lines = get_lines(header_file)[1:]
    num_leads = int(get_leads_num(header_file))
    baseline = np.zeros(num_leads)
    counter = 0
    for line in lines:
        line = line.split()
        counter += 1
        if counter > num_leads:
            break
        baseline[counter-1] = line[-3]
    return baseline


def features_padding(feature, max_length):
    processed_feature = []
    for data in feature:
        if data.shape[0] > max_length:
            data = data[:max_length]
        elif data.shape[0] < max_length:
            padding_shape = (max_length - data.shape[0],)
            padding = np.zeros(padding_shape, dtype=data.dtype)
            data = np.concatenate([data, padding], axis=0)
        processed_feature.append(data)
    return processed_feature


def extract_features(ecg_data, header, ecg_frequency, mean_age):
    """
    Extracts features from the ECG data.
    """
    detectors = Detectors(ecg_frequency)
    features = None
    try:
        peaks = np.array(detectors.pan_tompkins_detector(ecg_data[0]))
        age = 0
        try:
            age = int(get_patient_data(header)['Age'])
        except ValueError:
            age = mean_age
        sex = get_patient_data(header)['Sex']
        if sex == 'Male':
            sex = 0
        elif sex == 'Female':
            sex = 1
        else:
            sex = -1
        
        if np.any(peaks < 0):
            print("NEGATIVE PEAKS ", get_id(header))
            
            features = {
                'sex': np.array([sex]),
                'age': np.array([age]),
                'heart_rate': np.nan,
                'SDNN': np.nan,
                'RMSSD': np.nan,
                'pNNN50': np.nan
            }
        else:

            rr_intervals = np.diff(peaks) / ecg_frequency

            rr_intervals_no_outliers = remove_outliers(rr_intervals, verbose=False)
            nn_intervals = interpolate_nan_values(rr_intervals_no_outliers)

            time_domain_features = get_time_domain_features(nn_intervals)

            heart_rate =  time_domain_features['mean_hr']
            sdnn = time_domain_features['sdnn']
            rmssd = time_domain_features['rmssd']
            pNNN50 = time_domain_features['pnni_50']

            features = {
                'sex': np.array([sex]),
                'age': np.array([age]),
                'heart_rate': np.array([heart_rate]),
                'SDNN': np.array([sdnn]),
                'RMSSD': np.array([rmssd]),
                'pNNN50': np.array([pNNN50])
            }

    except Exception as e:
        print("FEATURES ERROR header:", get_id(header))
        print("Error message:", e)
        features = {}

    return features