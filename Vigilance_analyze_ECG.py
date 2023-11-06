#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28

Script that analyzes ECG features related to the vigilance experiement.

@author: nicolaiwolpert
"""
import numpy as np
import pandas as pd
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import norm
from scipy.integrate import simps
from collections import Counter
import glob
import warnings
warnings.filterwarnings("ignore")

from Scripts.vigilance_tools import *

import neurokit2 as nk

#%matplotlib

plt.close('all')

root = os.path.dirname(os.getcwd())
root = os.path.join(root, 'Vigilance')

raw_dir = os.path.join(root, 'Data_raw')
eeg_dir = os.path.join(raw_dir, 'EEG')
behavior_dir = os.path.join(raw_dir, 'Behavior')
data_work_path = root + '/Data_work/'

# Specify subjects for the different tasks with clean EEG & behavior
tasks_subjects = {'atc': ['01', '03', '04', '05', '08', '15', '21', '23', '24'],
                  'line_task_sim': ['02', '03', '04', '18'],
                  'line_task_succ': ['02', '04', '05', '07'],
                  'oddball': ['04', '07']}

SAMPLE_RATE = 250
codes_triggers = {"START_BLOCK": 10,"END_BLOCK": 11,
                  "START_TRIAL": 20,"END_TRIAL": 21,
                  "BUTTON_PRESS": 7, "ONSET_TARGETS": 30, "OFFSET_TARGETS": 31,
                  "ONSET_NONTARGETS": [40, 50], "OFFSET_NONTARGETS": [41, 51]
                  }

questionnaires = pd.read_csv(raw_dir + '/questionnaires_answers.csv', sep=';', dtype={'Subject ID': str})
questionnaires = questionnaires.dropna().reset_index(drop=True)
questionnaires["subject_id"] = questionnaires["subject_id"].astype(str).str.zfill(2)

# Parameters of the sliding window to compute blink rate
sliding_window_duration = 60        # in seconds
sliding_window_overlap = 0.5

# specify ECG features of interest
ecg_features_interest = ['ECG_Rate_Mean', 'HRV_MedianNN', 'HRV_HF', 'HRV_LF', 'HRV_VHF', 'HRV_LFHF']

######################################## Compute ECG features for each subject #########################################

print(('########## Computing ECG features for each subject.... ##########'))
ECG_features_by_subject = pd.DataFrame(columns=['task', 'subject_id', 'bin']+ecg_features_interest)
ECG_features_by_subject_bin = pd.DataFrame(columns=['task', 'subject_id', 'bin']+ecg_features_interest)
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        print(f'### {task}, S{subject_id}...')

        results_behavior = read_behavior_files(subject_id, task, raw_dir, block_type='experiment')

        filepath_ecg_csv = data_work_path + f'ECG_raw_notch/{task}_S{subject_id}/{task}_S{subject_id}_ecg_notch.csv'
        ecg_csv = pd.read_csv(filepath_ecg_csv, index_col=0)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(ecg_csv.loc[ecg_csv.trigger==codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(ecg_csv.loc[ecg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(ecg_csv.loc[ecg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(ecg_csv.loc[ecg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]

        # Set first timestamp to zero
        ecg_block = ecg_csv.loc[ind_start_exp:ind_end_exp].reset_index()
        ecg_block['timestamp'] -= ecg_block['timestamp'][0]
        ecg_block.set_index('timestamp', inplace=True)

        # For overall recording
        df_raw, info = nk.ecg_process(ecg_block['ECG'], sampling_rate=SAMPLE_RATE)
        ecg_features = nk.ecg_analyze(df_raw, sampling_rate=SAMPLE_RATE)
        ecg_features = ecg_features[ecg_features_interest]
        ecg_features = pd.concat(
            [pd.DataFrame({'task': [task], 'subject_id': [subject_id], 'bin': [bin]}), ecg_features], axis=1)

        ECG_features_by_subject = pd.concat([ECG_features_by_subject, ecg_features], axis=0)

        # Bin-wise: Divide the block into 4 bins based on trials to compute ECG indeces
        ind_starts_trials = list(ecg_block.loc[ecg_block.trigger == codes_triggers["START_TRIAL"]].index)
        ind_ends_trials = list(ecg_block.loc[ecg_block.trigger == codes_triggers["END_TRIAL"]].index)
        if len(ind_ends_trials) == 0:  # In line task, end of trial triggers were not sent, note them from start trial triggers
            warnings.warn('Found no triggers for end of trials. Reconstructing from start trial triggers')
            ind_ends_trials = [i - 1 for i in ind_starts_trials[1:]] + [
                ecg_block.loc[ecg_block.trigger == codes_triggers["END_BLOCK"]].index[-1]]

        ntrials = results_behavior.shape[0]
        ntrials_per_bin = distribute_items(ntrials, 4)
        trials_bin1 = [0, ntrials_per_bin[0] - 1]
        trials_bin2 = [ntrials_per_bin[0], ntrials_per_bin[0] + ntrials_per_bin[1] - 1]
        trials_bin3 = [ntrials_per_bin[0] + ntrials_per_bin[1],
                       ntrials_per_bin[0] + ntrials_per_bin[1] + ntrials_per_bin[2] - 1]
        trials_bin4 = [ntrials_per_bin[0] + ntrials_per_bin[1] + ntrials_per_bin[2],
                       ntrials_per_bin[0] + ntrials_per_bin[1] + ntrials_per_bin[2] + ntrials_per_bin[3] - 1]

        for bin in range(4):

            if bin == 0:
                ecg_signal = ecg_block.loc[ind_starts_trials[0]:ind_ends_trials[trials_bin1[1]]]['ECG']
            elif bin == 1:
                ecg_signal = ecg_block.loc[ind_starts_trials[trials_bin2[0]]:ind_ends_trials[trials_bin2[1]]]['ECG']
            elif bin == 2:
                ecg_signal = ecg_block.loc[ind_starts_trials[trials_bin3[0]]:ind_ends_trials[trials_bin3[1]]]['ECG']
            elif bin == 3:
                ecg_signal = ecg_block.loc[ind_starts_trials[trials_bin4[0]]:ind_ends_trials[trials_bin4[1]]]['ECG']

            df_raw, info = nk.ecg_process(ecg_signal, sampling_rate=SAMPLE_RATE)
            ecg_features = nk.ecg_analyze(df_raw, sampling_rate=SAMPLE_RATE)
            ecg_features = ecg_features[ecg_features_interest]
            for f in ecg_features_interest:
                if type(ecg_features[f][0]) == np.ndarray:
                    ecg_features[f] = ecg_features[f][0][0][0]
            ecg_features = pd.concat([pd.DataFrame({'task': [task], 'subject_id': [subject_id], 'bin': [bin]}), ecg_features], axis=1)

            ECG_features_by_subject_bin = pd.concat([ECG_features_by_subject_bin, ecg_features], axis=0)

ECG_features_by_subject_bin.to_csv(data_work_path + 'ECG_features_by_subject_bin.csv')

### Show subject-by-subject

for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        fig, axs = plt.subplots(int(len(ecg_features_interest)/2), 2)
        axs = axs.flatten()
        for ifeature, feature in enumerate(ecg_features_interest):
            ECG_feat_by_bin = [ECG_features_by_subject_bin.loc[(ECG_features_by_subject_bin.task == task) &
                                                               (ECG_features_by_subject_bin.subject_id == subject_id) &
                                                               (ECG_features_by_subject_bin.bin == bin), feature][0] for bin in range(4)]
            axs[ifeature].plot([1,2,3,4], ECG_feat_by_bin, label=task + '_' + subject_id)
            axs[ifeature].set_xlabel('bin')
            axs[ifeature].set_ylabel(feature)
        plt.suptitle(task + ', ' + subject_id)
        plt.tight_layout()

### Show all in one

normalize = True

# Get maximally distinguishable colors
colors = matplotlib.cm.tab20(range(int(ECG_features_by_subject_bin.shape[0]/4)))
for ifeature, feature in enumerate(ecg_features_interest):

    plt.figure()
    itask_subject = 0
    for task in tasks_subjects.keys():
        for subject_id in tasks_subjects[task]:
            ECG_feat_by_bin = [ECG_features_by_subject_bin.loc[(ECG_features_by_subject_bin.task == task) &
                                                               (ECG_features_by_subject_bin.subject_id == subject_id) &
                                                               (ECG_features_by_subject_bin.bin == bin), feature][0] for bin in range(4)]

            if normalize:
                ECG_feat_by_bin = list((ECG_feat_by_bin - np.min(ECG_feat_by_bin)) / (np.max(ECG_feat_by_bin) - np.min(ECG_feat_by_bin)))
            plt.plot([1,2,3,4], ECG_feat_by_bin, color = colors[itask_subject], label=task + '_' + subject_id)
            plt.scatter([1,2,3,4], ECG_feat_by_bin, color = colors[itask_subject])
            itask_subject += 1
    plt.xlabel('bin')
    plt.ylabel(feature)
    if ifeature == len(ecg_features_interest): plt.legend();
    plt.show()


