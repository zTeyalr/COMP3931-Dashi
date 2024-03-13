import scipy.io as sc
import wfdb
import numpy as np
import os
import data_loader
def main():
    sample_data = 'sample_data'
    counter = 0
    for filename in os.listdir(sample_data):
        if filename.endswith('.mat'):
            counter+=1
            
            mat_file = os.path.join(sample_data, filename)
            hea_file = mat_file.replace('.mat', '.hea')

            try:
                ecg_dict = data_loader.load_mat(mat_file)
                metadata = data_loader.load_header(hea_file)

                print("ECG Data:", ecg_dict)
                print("Metadata:", metadata)

            except FileNotFoundError:
                print(f"Warning: Corresponding file not found for {filename}")
    
    print(counter)
    return

if __name__ == "__main__":
    main()