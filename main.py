import scipy.io as sc
import wfdb
import numpy as np
import os
import data_loader
import preprocessing
import matplotlib.pyplot as plt

def main():
    sample_data = 'sample_data\WFDB_Ga'
    counter = 0
    
    for filename in os.listdir(sample_data):
        if filename.endswith('.mat'):
            counter+=1
            if counter > 1:
                break
            
            mat_file = os.path.join(sample_data, filename)
            hea_file = mat_file.replace('.mat', '.hea')

            try:
                ecg_dict = data_loader.load_mat(mat_file)
                metadata = data_loader.load_header(hea_file)
                freq=data_loader.get_frequency(metadata)
                
                # Preprocessing the data
                ecg_pre = preprocessing.butterworth_highpass(ecg_dict['ecg_signal'], cutoff=10, ecg_frequency=freq)
                ecg_pre = preprocessing.butterworth_bandpass(ecg_pre, [0.5, 40], freq) # 1hz and 45hz are standard in
                ecg_pre, outliers = preprocessing.z_normalize(ecg_pre)
                ecg_pre = preprocessing.zero_padding(ecg_pre)
               
                # Plot the data to visualize before and after preprocessing
                
                # time_val = np.arange(ecg_dict['ecg_signal'].shape[1]) / data_loader.get_frequency(metadata)
                # time_val2 = np.arange(ecg_pre.shape[1]) / data_loader.get_frequency(metadata)
                # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,8), sharex=True)

                # for i in range(12):
                #     axes[0].plot(time_val, ecg_dict['ecg_signal'][i])
                #     #axes[0].scatter(time_val[outliers[i,:]], ecg_dict['ecg_signal'][i, outliers[i, :]], color='red', marker='x')
                #     axes[0].set_title(f'Original ECG Signal - {filename}')
                #     axes[0].set_ylabel('Amplitude')
                # for i in range(12):
                #     # # Plot preprocessed signal
                #     axes[1].plot(time_val2, ecg_pre[i])
                #     axes[1].set_title(f'Preprocessed ECG Signal - {filename}')
                #     axes[1].set_xlabel('Time (seconds)')
                #     axes[1].set_ylabel('Amplitude')

                # plt.xlabel('Time (seconds)')
                # plt.ylabel('Amplitude')
                # plt.title('ECG Signal - All Leads Overlayed')
                # plt.tight_layout()
                # plt.show()

            except FileNotFoundError:
                print(f"Warning: Corresponding file not found for {filename}")
    
    print(counter)
    return

if __name__ == "__main__":
    main()