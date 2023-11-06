#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:54:58 2022

Script that creates raw fif files for eeg and ecg and notch-filters data.

@author: nicolaiwolpert
"""

import os.path
import mne
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
import pandas as pd
import matplotlib.pyplot as plt

root_dir = 'C:/Users/nwolpert/Documents/Mind&Act/Vigilance/'
data_raw_path = root_dir + '/Data_raw/'
data_work_path = root_dir + '/Data_work/'

notchfilter = True

# Specify sample frequency and low-/high bandpass frequencies
sfreq = 250
FILTER_LOW_PASS = 1
FILTER_HIGH_PASS = 35

# Specify task ('oddball', 'atc', 'line_task_sim', or 'line_task_succ')
task = 'line_task_sim'

# Specify subject ID
subject_id = '30'

# read csv
filepath_eeg = f'{data_raw_path}{task}/EEG/{subject_id}/vigilance_{task}_EEG_{subject_id}.csv'
if not os.path.isfile(filepath_eeg):
    raise ValueError(f'{filepath_eeg} not found')

alldata = pd.read_csv(filepath_eeg, low_memory=False, index_col=0)
if "P4" in alldata.columns:        # After first couple of subjects, EEG channels were named correctly from the beginning
    alldata.rename(columns={"O1": "Pz", "O2": "O1", "Pz": "O2", "CPz": "ECG", "P3": "F3", "P4": "F4", "F3": "Fp1", "F4": "Fp2"}, inplace=True)
else:
    alldata.rename(columns={"Fz": "ECG"}, inplace=True)
data_original = alldata.copy()

# transform to raw format
d = alldata.copy()
ch_names = list(d.columns.values)
# ECG has to be treated like EEG channel, else notch filter doesn't work
ch_types = ['eeg'] * 8 + ['stim']
dValues = d.values.T

infor = create_info(ch_names, sfreq, ch_types)
eeg_raw = RawArray(dValues, info=infor, verbose=False)
if notchfilter:
    print('Notch-filtering...')
    eeg_raw.notch_filter(freqs=[50, 100], filter_length='auto', method='fir', notch_widths=None,
                               trans_bandwidth=1.0, n_jobs=1, iir_params=None, mt_bandwidth=None, p_value=0.05,
                               phase='zero', fir_window='hamming', fir_design='firwin', pad='reflect_limited',
                               verbose=None)
    eeg_raw.filter(1, 35, fir_design='firwin')

# Note filtered data in dataframe to save them as csv later
channel_names = eeg_raw.info.ch_names
for channel in channel_names:
    if len(eeg_raw[channel][0][0])!=alldata.shape[0]:
        raise ValueError('Number of samples between fif and csv file does not match for subject' + subject_id)
    alldata[channel] = eeg_raw[channel][0][0]

# Seperate EEG from ECG
# ECG fif
ecg_raw = eeg_raw.copy().pick_channels(set(['ECG']))
if notchfilter:
    path_save = data_work_path + f'ECG_raw_notch/{task}_S{subject_id}/'
else:
    path_save = data_work_path + f'ECG_raw/{task}_S{subject_id}/'
if not os.path.exists(path_save):
    os.makedirs(path_save)
if notchfilter:
    ecg_raw.save(path_save + f'{task}_S{subject_id}_ecg_notch_raw.fif', overwrite=True)
else:
    ecg_raw.save(path_save + f'{task}_S{subject_id}_ecg_raw.fif', overwrite=True)

# ECG csv
ecg_csv = alldata[['ECG', 'trigger']]
if not len(ecg_raw['ECG'][0][0])==ecg_csv.shape[0]:
    raise ValueError('Number of samples for ECG in csv and fif does not correspond')
if notchfilter:
    ecg_csv.to_csv(path_save + f'{task}_S{subject_id}_ecg_notch.csv')
else:
    ecg_csv.to_csv(path_save + f'{task}_S{subject_id}_ecg.csv')

# EEG fif
eeg_raw = eeg_raw.pick_channels(list(set(eeg_raw.info.ch_names) - set(['ECG'])))
eeg_raw.set_montage(make_standard_montage('standard_1020'))
if notchfilter:
    path_save = data_work_path + f'EEG_raw_notch/{task}_S{subject_id}/'
else:
    path_save = data_work_path + f'EEG_raw/{task}_S{subject_id}/'
if not os.path.exists(path_save):
    os.makedirs(path_save)
if notchfilter:
    eeg_raw.save(path_save + f'{task}_S{subject_id}_eeg_notch_raw.fif', overwrite=True)
else:
    eeg_raw.save(path_save + f'{task}_S{subject_id}_eeg_raw.fif', overwrite=True)

# EEG csv
eeg = alldata.copy()
eeg.drop('ECG', axis=1, inplace=True)
if notchfilter:
    eeg.to_csv(path_save + f'{task}_S{subject_id}_eeg_notch.csv')
else:
    eeg.to_csv(path_save + f'{task}_S{subject_id}_eeg.csv')

print('Done.')


# Optional: Visualize selected
eeg_raw = mne.io.read_raw_fif(data_work_path + f'EEG_raw_notch/{task}_S{subject_id}/{task}_S{subject_id}_eeg_notch_raw.fif', preload=True)
ecg_raw = mne.io.read_raw_fif(data_work_path + f'ECG_raw_notch/{task}_S{subject_id}/{task}_S{subject_id}_ecg_notch_raw.fif', preload=True)

plt.close('all')

channel = 'Pz'

fig, axs = plt.subplots()
axs.plot(eeg_raw[channel][1][2000:], eeg_raw[channel][0][0][2000:])
axs.set_xlabel('seconds')
axs.set_title('subject' + subject_id + ', ' + task + ', ' + channel + ', notch-filtered')
plt.show()

fig, axs = plt.subplots()
axs.plot(ecg_raw['ECG'][1][2000:], ecg_raw['ECG'][0][0][2000:])
axs.set_xlabel('seconds')
axs.set_title('subject' + subject_id + ', ' + task + ', ECG notch-filtered')
plt.show()
