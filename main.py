import os
import tarfile

import numpy as np
import scipy.io as sc
import wfdb
import matplotlib.pyplot as plt

import data_loader
import preprocessing
import argparse
from training import train_model
import warnings

def main():
    """main method"""
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Preprocessing or training ECG model")
    parser.add_argument("mode", choices=["preprocess", "train"], help="Select mode: 'preprocess' or 'train'")
    if 'train' in parser.parse_known_args()[0].mode: 
        parser.add_argument("--num_workers", type=int, default=0, help="Select the number of workers for DataLoader (default: 0)")

    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocessing.preprocess_and_save('sample_data', 'preprocessed_data')
        #data_loader.save_preprocessed('sample_data', 'preprocessed_data', 500, preprocessing.preprocess_data)
    elif args.mode == "train":
        train_model(num_workers=args.num_workers)


    return


if __name__ == "__main__":
    main()