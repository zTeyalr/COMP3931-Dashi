import os
import tarfile

import numpy as np
import scipy.io as sc
import wfdb

import matplotlib.pyplot as plt

import data_loader

def main():
    """main method"""
    sample_data = "sample_data\WFDB_Ga"
    n = 10
    i = 0
    for filename in os.listdir(sample_data):
        i += 1
        if i > 10:
            break
        if filename.endswith('.mat'):
            
            mat_file = os.path.join(sample_data, filename)
            hea_file = mat_file.replace('.mat', '.hea')

            try:
                ecg_dict = data_loader.load_mat(mat_file)
                metadata = data_loader.load_header(hea_file)
                #print(ecg_dict)

            except FileNotFoundError:
                print(f"Warning: Corresponding file not found for {filename}")
    return


if __name__ == "__main__":
    main()