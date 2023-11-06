#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:54:58 2022

Script that rejects artifacts in EEG data.
This script is ONLY applied on notch-filtered data, and never used for non-notchfiltered data that are used for Deep
Learning.

Saves:
- Data cleaned from artifacts only, in EEG_per_block_raw_notch_cleaned_art
- Data cleaned from artifacts and response windows, in EEG_per_block_raw_notch_cleaned_art_resp_win

@author: nicolaiwolpert
"""

import os.path
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne import create_info, concatenate_raws
from mne.channels import make_standard_montage
from mne.io import RawArray
import warnings
import asrpy
warnings.filterwarnings("ignore")
#%matplotlib

from Scripts.vigilance_tools import transform_raw

plt.close('all')

root = os.path.dirname(os.getcwd())
root_dir = root + '/Vigilance/'
data_raw_path = root_dir + '/Data_raw/'
data_work_path = root_dir + '/Data_work/'

# Specify task ('oddball', 'atc', 'line_task_sim', or 'line_task_succ')
task = 'oddball'

# Specify subject ID
subject_id = '07'

# Select artefact rejection method: 'threshold' or 'asr'
method_artrej = 'asr'

# Specify absolute threshold for normal signal range
thresh_signal_range = 100

# Specify sample frequency
SFREQ = 250

# Specify artifact padding in seconds
padding_amplitude = 1
padding_amplitude_samples = int(padding_amplitude*SFREQ)

# Trigger codes for participant response to query and rating screen
TRIGGER_RESPONSE_SUBJECT = 7

# Choose padding around button presses (in seconds)
reject_button_presses = False
padding_button_presses = 1
padding_button_presses_samples = padding_button_presses*SFREQ

# This is the size for the window to estimate PSD using Welch's method
psd_window_size_seconds = 4
psd_overlap = 0.5

### Helper functions

def find_start_end_artifacts(indeces_artifacts):
    '''
    Find the start and end sample of each artifact
    :param indeces_artifacts: indeces of the artifacts
    :return:
    '''

    indeces_artifacts.sort()
    start = end = indeces_artifacts[0]
    samples_start_end = []
    for i in range(1, len(indeces_artifacts)):
        if indeces_artifacts[i] == indeces_artifacts[i-1]+1:
            end = indeces_artifacts[i]
        else:
            samples_start_end.append([start, end])
            start = end = indeces_artifacts[i]
    samples_start_end.append([start, end])

    return samples_start_end

def reject_artifacts_amplitude(data, artifacts, padding_amplitude_samples):
    '''
    Rejects artifacts in fif data
    :param data:
    :param idx_channel:
    :param artifacts:
    :param padding_amplitude_samples:
    :return: data_clean
    '''
    data_clean = data.copy()
    for artifact in artifacts:
        # add padding
        artifact_padded = artifact.copy()
        # avoid getting out of range
        if artifact_padded[0] - padding_amplitude_samples > 0:
            artifact_padded[0] = artifact_padded[0] - padding_amplitude_samples
        else:
            artifact_padded[0] = 0
        if artifact_padded[1] + padding_amplitude_samples < len(data_clean):
            artifact_padded[1] = artifact_padded[1] + padding_amplitude_samples
        else:
            artifact_padded[1] = len(data_clean)

        data_clean[artifact_padded[0]:artifact_padded[1]] = np.nan

    return data_clean

### Load files

## EEG
# read csv
filepath_csv = data_work_path + f'EEG_raw_notch/{task}_S{subject_id}/{task}_S{subject_id}_eeg_notch.csv'
eeg_csv = pd.read_csv(filepath_csv, low_memory=False, index_col=0)
channel_names = list(eeg_csv.drop('trigger', axis=1).columns)

# temporarily reset index
eeg_csv.reset_index(inplace=True)
eeg_orig = eeg_csv.copy()

# read fif
filepath_fif = data_work_path + f'EEG_raw_notch/{task}_S{subject_id}/{task}_S{subject_id}_eeg_notch_raw.fif'
if os.path.isfile(filepath_fif):
    eeg_fif = mne.io.read_raw_fif(filepath_fif, preload=True)
else:
    raise ValueError('Subject' + subject_id + ', corresponding fif file not found')
eeg_fif_orig = eeg_fif.copy()

if eeg_csv.shape[0] != eeg_fif._data.shape[1]:
    raise ValueError('Number of samples does not match between csv and fif')
nsamples_original = eeg_fif._data.shape[1]

## ECG
filepath_ecg = data_work_path + f'ECG_raw_notch/{task}_S{subject_id}/{task}_S{subject_id}_ecg_notch.csv'
ecg = pd.read_csv(filepath_ecg, low_memory=False, index_col=0)

### Note bad channels by visual inspection

filepath_log_bad_channels = data_work_path + f'/Artefacts/log_bad_channels_{task}_subject{subject_id}.csv'
if os.path.isfile(filepath_log_bad_channels):
    log_bad_channels = pd.read_csv(filepath_log_bad_channels, index_col=0)
else:
    log_bad_channels = pd.DataFrame(columns={'subject','channel'})
    eeg_raw = eeg_csv.copy()
    eeg_raw = eeg_raw.iloc[2000:]    # ignore beginning, often big peak in the start
    eeg_raw = transform_raw(eeg_raw)
    for ichan, channel in enumerate(channel_names):
        fig = plt.figure(figsize=(20, 5))
        plt.plot(eeg_csv.iloc[2000:][channel])
        plt.title('Subject' + subject_id + ', channel ' + channel, fontsize=15)
        plt.show()

        window_size_samples = int(psd_window_size_seconds * SFREQ)
        samples_overlap = int(window_size_samples / 2)
        psds, freqs = mne.time_frequency.psd_welch(eeg_raw, picks='eeg', fmin=1, fmax=30, n_fft=window_size_samples, n_overlap=samples_overlap)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(freqs, psds[ichan])
        plt.xlabel('Frequency')
        plt.title('Subject' + subject_id + ', channel ' + channel, fontsize=15)
        plt.show()

        plt.pause(1)

        key = input("Bad channel? Press <b>, else press any other key")
        if key=='b':
            print('Noting', channel, 'as bad channel')
            if log_bad_channels.loc[(log_bad_channels.subject==subject_id) & (log_bad_channels.channel==channel)].empty:
                log_bad_channels = log_bad_channels.append(pd.DataFrame({'subject': [subject_id], 'channel': [channel]}))
        plt.close()
        plt.close()

    if not os.path.exists(data_work_path + f'/{task}/Artefacts/'):
        os.makedirs(data_work_path + f'/{task}/Artefacts/')
    log_bad_channels.reset_index(inplace=True, drop=True)
    log_bad_channels.to_csv(data_work_path + f'Artefacts/log_bad_channels_{task}_subject{subject_id}.csv')

'''
# Optional: visualize for control
fig, axs = plt.subplots(int(np.ceil(len(channel_names) / 2)), 2, figsize=(10, 5), sharex=True)
axs = axs.flatten()
for ichan, channel in enumerate(channel_names):
    axs[ichan].plot(eeg_orig['timestamp'], eeg_orig[channel], 'r')
    axs[ichan].plot(eeg_csv['timestamp'], eeg_csv[channel])
    axs[ichan].set_title('Subject' + subject_id + ', ' + channel + ' after removing button presses', fontsize=15)
plt.tight_layout()
plt.show()
'''


### Artefact rejection

if method_artrej == 'threshold':
    # Reject samples with abnormally high signal based on threshold
    # Not doing this for channels Fp1/Fp2 because blinks exceed 100 mV easily

    channel_names = eeg_fif.copy().info.ch_names
    channel_names.remove('trigger')
    for channel in channel_names:

        # reject whole signal if bad channel
        if not log_bad_channels.loc[log_bad_channels.channel==channel].empty:
            print('Bad channel, rejecting whole signal')
            eeg_csv[channel] = np.nan
            idx_channel = eeg_fif.info.ch_names.index(channel)
            eeg_fif._data[idx_channel] = np.nan
        else:
            ### For Fp1/Fp2 reject at least massive artifactss
            if channel in ['Fp1', 'Fp2']:
                thresh = 500
            else:
                thresh = thresh_signal_range

            # Safety check: if there are artifacts in fif file there should also be in csv and vice versa
            artifact_fif = False

            # ...in fif file
            idx_channel = eeg_fif.info.ch_names.index(channel)
            idx_outrange = abs(eeg_fif._data[idx_channel]) > thresh
            idx_outrange = [i for i, x in enumerate(idx_outrange) if x==True]

            if idx_outrange:
                artifact_fif = True
                artifacts = find_start_end_artifacts(idx_outrange)
                eeg_fif._data[idx_channel] = reject_artifacts_amplitude(eeg_fif._data[idx_channel], artifacts, padding_amplitude_samples)

            # ...in csv file
            # Safety check: if there are artifacts in fif, there should be in csv
            if artifact_fif and not any(abs(eeg_csv[channel]) > thresh_signal_range):
                raise ValueError('Subject' + subject_id + ', channel ' + channel + ', found artifact in fif but not csv')

            if artifact_fif:
                eeg_csv[channel] = reject_artifacts_amplitude(eeg_csv[channel], artifacts, padding_amplitude_samples)


    ### Visualize for control
    # One channel
    channel_plot = 'Pz'
    fig, axs = plt.subplots(2,1,sharex=True)
    axs = axs.flatten()
    axs[0].plot(eeg_orig.timestamp, eeg_orig[channel_plot])
    axs[0].set_title(task + ', subject ' + subject_id + ', channel ' + channel_plot + ', data original')
    axs[1].plot(eeg_csv.timestamp, eeg_csv[channel_plot])
    axs[1].set_xlabel('seconds')
    axs[1].set_title(task + ', subject ' + subject_id + ', channel ' + channel_plot + ' after rejecting signal > ' + str(thresh_signal_range))
    plt.show()

    # All channels
    fig, axs = plt.subplots(int(np.ceil(len(channel_names) / 2)), 2, figsize=(10, 5), sharex=True)
    axs = axs.flatten()
    for ichan, channel in enumerate(channel_names):
        axs[ichan].plot(eeg_orig['timestamp'], eeg_orig[channel], 'r')
        axs[ichan].plot(eeg_csv['timestamp'], eeg_csv[channel])
        axs[ichan].set_ylim([-200, 200])
        axs[ichan].set_title(task + ', subject' + subject_id + ', ' + channel + ' cleaned',
                             fontsize=15)
    plt.tight_layout()
    plt.show()

elif method_artrej == 'asr':

    asr = asrpy.ASR(SFREQ, cutoff=20)
    asr.fit(eeg_fif)
    eeg_fif = asr.transform(eeg_fif)

    for ichannel, channel in enumerate(channel_names):
        eeg_csv[channel] = eeg_fif._data[ichannel]

    # reject whole signal if bad channel
    for channel in channel_names:
        if not log_bad_channels.loc[log_bad_channels.channel == channel].empty:
            print('Bad channel, rejecting whole signal')
            eeg_csv[channel] = np.nan
            idx_channel = eeg_fif.info.ch_names.index(channel)
            eeg_fif._data[idx_channel] = np.nan

    '''
    # Reject remaining portions with aberrant range
    for channel in channel_names:

        # reject whole signal if bad channel
        if log_bad_channels.loc[log_bad_channels.channel==channel].empty:

            ### For Fp1/Fp2 reject at least massive artifactss
            if channel in ['Fp1', 'Fp2']:
                thresh = 500
            else:
                thresh = thresh_signal_range

            # Safety check: if there are artifacts in fif file there should also be in csv and vice versa
            artifact_fif = False

            # ...in fif file
            idx_channel = eeg_fif.info.ch_names.index(channel)
            idx_outrange = abs(eeg_fif._data[idx_channel]) > thresh
            idx_outrange = [i for i, x in enumerate(idx_outrange) if x==True]

            if idx_outrange:
                artifact_fif = True
                artifacts = find_start_end_artifacts(idx_outrange)
                eeg_fif._data[idx_channel] = reject_artifacts_amplitude(eeg_fif._data[idx_channel], artifacts, padding_amplitude_samples)

            # ...in csv file
            # Safety check: if there are artifacts in fif, there should be in csv
            if artifact_fif and not any(abs(eeg_csv[channel]) > thresh_signal_range):
                raise ValueError('Subject' + subject_id + ', channel ' + channel + ', found artifact in fif but not csv')

            if artifact_fif:
                eeg_csv[channel] = reject_artifacts_amplitude(eeg_csv[channel], artifacts, padding_amplitude_samples)
    '''

    # Visualize
    for channel in channel_names:
        if any(abs(eeg_csv[channel]) > thresh_signal_range):
            print(f'Still found signal in aberrant range for {task}, subject {subject_id}, channel {channel}')
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        axs = axs.flatten()
        axs[0].plot(eeg_orig['timestamp'], eeg_orig[channel])
        axs[0].set_title(f'Channel {channel} original')
        axs[0].set_ylim([-200, 200])
        axs[1].plot(eeg_csv['timestamp'], eeg_csv[channel])
        axs[1].set_title(f'Channel {channel} after ARS artefact correction')
        axs[1].set_xlabel('time (seconds)')
        axs[1].set_ylim([-200, 200])

else:
    raise ValueError('Invalid artefact rejection method specified')

if eeg_csv.shape[0] != nsamples_original:
    raise ValueError('CSV file does not have original sample number after artifact rejection anymore')
if eeg_fif._data.shape[1] != nsamples_original:
    raise ValueError('fif file does not have original sample number after artifact rejection anymore')

### Save csv and fif cleaned from artifacts
if not os.path.exists(data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/'):
    os.makedirs(data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/')
eeg_csv.to_csv(
    data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv', index=False)
eeg_fif.save(
    data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned_raw.fif',
    overwrite=True)

### Show PSD after cleaning from artifacts

filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
filepath_eeg_fif = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned_raw.fif'
eeg_csv = pd.read_csv(filepath_eeg_csv)
eeg_fif = mne.io.read_raw_fif(filepath_eeg_fif, preload=True)

window_size_samples = int(psd_window_size_seconds * SFREQ)
samples_overlap = int(window_size_samples / 2)
psds, freqs = mne.time_frequency.psd_welch(eeg_fif.crop(tmin=600, tmax=2500), picks='eeg', fmin=1, fmax=30, n_fft=window_size_samples,
                                           n_overlap=samples_overlap)

fig, axs = plt.subplots(1, 3)
axs = axs.flatten()
axs[0].plot(freqs, np.log(psds[1]))
axs[0].set_xlabel('frequency (Hz)')
axs[0].set_ylabel('log power')
axs[0].set_title('O1')
axs[1].plot(freqs, np.log(psds[0]))
axs[1].set_xlabel('frequency (Hz)')
axs[1].set_title('Pz')
axs[2].plot(freqs, np.log(psds[2]))
axs[2].set_xlabel('frequency (Hz)')
axs[2].set_title('O2')
plt.suptitle(task + ', S' + subject_id)
plt.show()


### Replace bad channels with neighbouring channels
# For example: If channel F3 bad, replace with time series from channel F4. If F3 and F4 are bad, replace with signal
# from Fp1 and Fp2
# Do this on both raw and notch_filtered&artifeact-cleaned data

filepath_raw = data_raw_path + f'{task}/EEG/{subject_id}/vigilance_{task}_EEG_{subject_id}.csv'
eeg_raw = pd.read_csv(filepath_csv, low_memory=False, index_col=0)

filepath_notch_artcleaned = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
eeg_notch_artcleaned = pd.read_csv(filepath_notch_artcleaned, low_memory=False, index_col=0)

log_bad_channels.reset_index(inplace=True, drop=True)
bad_channels = list(log_bad_channels.channel)
for ibad in list(range(log_bad_channels.shape[0])):

    bad_channel = log_bad_channels.loc[ibad, 'channel']

    if bad_channel == 'O1':
        eeg_raw['O1'] = eeg_raw['Pz']
        eeg_notch_artcleaned['O1'] = eeg_notch_artcleaned['Pz']
    elif bad_channel == 'O2':
        eeg_raw['O2'] = eeg_raw['Pz']
        eeg_notch_artcleaned['O2'] = eeg_notch_artcleaned['Pz']
    elif bad_channel == 'F3':
        if 'F4' not in bad_channels:
            eeg_raw['F3'] = eeg_raw['F4']
            eeg_notch_artcleaned['F3'] = eeg_notch_artcleaned['F4']
        else:
            eeg_raw['F3'] = eeg_raw['Fp1']
            eeg_notch_artcleaned['F3'] = eeg_notch_artcleaned['Fp1']
    elif bad_channel == 'F4':
        if 'F3' not in bad_channels:
            eeg_raw['F4'] = eeg_raw['F2']
            eeg_notch_artcleaned['F4'] = eeg_notch_artcleaned['F2']
        else:
            eeg_raw['F4'] = eeg_raw['Fp2']
            eeg_notch_artcleaned['F4'] = eeg_notch_artcleaned['Fp2']
    elif bad_channel == 'Fp1':
        if 'Fp2' not in bad_channels:
            eeg_raw['Fp1'] = eeg_raw['Fp2']
            eeg_notch_artcleaned['Fp1'] = eeg_notch_artcleaned['Fp2']
        else:
            eeg_raw['Fp1'] = eeg_raw['F3']
            eeg_notch_artcleaned['Fp1'] = eeg_notch_artcleaned['F3']
    elif bad_channel == 'Fp2':
        if 'Fp1' not in bad_channels:
            eeg_raw['Fp2'] = eeg_raw['Fp1']
            eeg_notch_artcleaned['Fp2'] = eeg_notch_artcleaned['Fp1']
        else:
            eeg_raw['Fp2'] = eeg_raw['F4']
            eeg_notch_artcleaned['Fp2'] = eeg_notch_artcleaned['F4']

if not os.path.exists(data_work_path + f'EEG_raw_replaced_bad_chans/'):
    os.mkdir(data_work_path + f'EEG_raw_replaced_bad_chans/')
if not os.path.exists(data_work_path + f'EEG_raw_replaced_bad_chans/{task}_S{subject_id}/'):
    os.mkdir(data_work_path + f'EEG_raw_replaced_bad_chans/{task}_S{subject_id}/')
    eeg_raw.to_csv(data_work_path + f'EEG_raw_replaced_bad_chans/{task}_S{subject_id}/{task}_S{subject_id}_eeg_replaced_bad.csv')
if not os.path.exists(data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}_replaced_bad_chans/'):
    os.mkdir(data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}_replaced_bad_chans/')
if not os.path.exists(data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}_replaced_bad_chans/{task}_S{subject_id}/'):
    os.mkdir(data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}_replaced_bad_chans/{task}_S{subject_id}/')
    eeg_notch_artcleaned.to_csv(data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}_replaced_bad_chans/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned_replaced_bad.csv')
