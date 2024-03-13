import scipy.io as sc
import wfdb
import numpy as np
import os

def load_mat(file_name):
    
    mat_data = sc.loadmat(file_name)
    ecg_signals = mat_data['val']

    return {'ecg_signal': ecg_signals}

def load_header(file_name):
    
    with open(file_name, 'r') as f:
        header = f.read()
    return header
    