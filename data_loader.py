""" LOAD DATA AND RETRIEVE INFORMATION
    Some of these function have been taken from the official
    Physionet Challenge 2021 Github helper_code.py template"""

import scipy.io as sc
import wfdb
import numpy as np
import os

# Load .mat files
def load_mat(mat_file):
    """Function loading mat files"""
    mat_data = sc.loadmat(mat_file)
    ecg_signals = mat_data['val']

    return {'ecg_signal': ecg_signals}

# Load .hea files
def load_header(header_file):
    """Function loading header files"""
    with open(header_file, 'r', encoding="utf-8") as f:
        header = f.read()
    return header


# Get info from header file as wfdb.rdsamp does not work with custom headers

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
        leads_num = get_line(header_file, 0).split(' ')[1]
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
                value = x[1].strip()
                
                if value.isdigit():
                    value = int(value)
                    
                patient_data[key] = value
                
            except IndexError:
                print(f"Out of bound Index")
            except ValueError:
                print(f"Error parsing")
            

    
    return patient_data