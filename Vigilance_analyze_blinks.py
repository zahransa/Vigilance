#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28

Script that analyzes blink rate over the course of the vigilance experiments.

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

import mne
from mne import create_info, concatenate_raws
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne import EpochsArray
from mne.datasets import sample
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

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

# Select z-threshold for blink detection
thresh_z_blinks = 2.5
padding_blinks = 0.1    # in seconds
padding_blinks_samples = padding_blinks * SAMPLE_RATE

# Parameters of the sliding window to compute blink rate
sliding_window_duration = 60        # in seconds
sliding_window_overlap = 0.5

################################## Show blink rate in moving windows for each subject ##################################

for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        eeg_csv = pd.read_csv(filepath_eeg_csv, index_col=0)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]

        # Set first timestamp to zero
        eeg_block = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index()
        eeg_block['timestamp'] -= eeg_block['timestamp'][0]
        eeg_block.set_index('timestamp', inplace=True)

        blinks_starts_ends_samples, blinks_timestamps = detect_blinks(eeg_block, thresh_z_blinks, padding_blinks_samples, task, subject_id, visualize=False)
        blinkrate_moving_windows = compute_blinkrate_sliding_window(blinks_timestamps, sliding_window_duration, sliding_window_overlap, block_duration=eeg_block.index[-1])

        # Add triggers of interest
        eeg_block = insert_triggers_of_interest(eeg_block, codes_triggers)
        ind_hits = eeg_block.loc[eeg_block.triggers_of_interest=='hit'].index
        ind_misses = eeg_block.loc[eeg_block.triggers_of_interest=='miss'].index
        ind_false_alarms = eeg_block.loc[eeg_block.triggers_of_interest=='false_alarm'].index

        fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(12,8))
        axs.flatten()
        axs[0].plot(blinkrate_moving_windows.center_window, blinkrate_moving_windows.blink_rate)
        # plot line of best fit
        a, b = np.polyfit(list(blinkrate_moving_windows.dropna().center_window), list(blinkrate_moving_windows.dropna().blink_rate), 1)
        axs[0].plot(blinkrate_moving_windows.center_window, a*blinkrate_moving_windows.center_window+b, color = 'grey', label='best fit blink rate')
        axs[0].set_xlabel('center window (seconds)')
        axs[0].set_ylabel('blink rate')
        axs[0].set_title(f'{task}, S{subject_id}, {sliding_window_duration} sec. moving window, {sliding_window_overlap*100}% overlap')
        middle_block = int(eeg_block.shape[0]/250 / 2)
        ylim = axs[0].get_ylim()
        vigilance_rating1 = questionnaires.loc[(questionnaires['task']==task) & (questionnaires['subject_id']==subject_id), 'NASA_vigilance1'].values[0]
        vigilance_rating2 = questionnaires.loc[(questionnaires['task']==task) & (questionnaires['subject_id']==subject_id), 'NASA_vigilance2'].values[0]
        numbers = np.arange(1, 21)
        colors_ratings = plt.cm.RdYlGn(numbers / 20.0)
        axs[0].text(middle_block - (middle_block/2), ylim[1]*1.1, 'vig. rat.=' + str(vigilance_rating1), color = colors_ratings[int(vigilance_rating1)], fontsize = 12, horizontalalignment='center')
        axs[0].text(middle_block + (middle_block/2), ylim[1]*1.1, 'vig. rat.=' + str(vigilance_rating2), color = colors_ratings[int(vigilance_rating2)], fontsize = 12, horizontalalignment='center')
        label_added = False
        for ind_hit in ind_hits:
            if not label_added:
                axs[0].scatter(ind_hit, np.nanmin(blinkrate_moving_windows['blink_rate']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='green', s=50, label='hits')
                label_added = True
            else:
                axs[0].scatter(ind_hit, np.nanmin(blinkrate_moving_windows['blink_rate']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='green', s=50)
        label_added = False
        for ind_miss in ind_misses:
            if not label_added:
                axs[0].scatter(ind_miss, np.nanmin(blinkrate_moving_windows['blink_rate']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='red', s=50, label='misses')
                label_added = True
            else:
                axs[0].scatter(ind_miss, np.nanmin(blinkrate_moving_windows['blink_rate']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='red', s=50)
        label_added = False
        for ind_false_alarm in ind_false_alarms:
            if not label_added:
                axs[0].scatter(ind_false_alarm, np.nanmin(blinkrate_moving_windows['blink_rate']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='orange', s=50, label='false alarms')
                label_added = True
            else:
                axs[0].scatter(ind_false_alarm, np.nanmin(blinkrate_moving_windows['blink_rate']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='orange', s=50)
        axs[0].set_ylim([ylim[0]*0.6, ylim[1] * 1.3])
        axs[0].legend(loc='center right')

        results_behavior = read_behavior_files(subject_id, task, raw_dir, block_type='experiment')
        # Show accuracy per bin
        performance_indeces_bin1, performance_indeces_bin2, performance_indeces_bin3, performance_indeces_bin4 = compute_performance_per_bin(
            results_behavior, task)
        def partition_and_calculate_midpoints(signal):
            N = len(signal)
            chunk_length = N // 4
            # Reshape the signal into a 2D array with 4 rows and chunk_length columns
            reshaped_signal = np.reshape(signal[:4 * chunk_length], (4, chunk_length))
            # Calculate the midpoints along the columns and flatten the result into a 1D array
            midpoints = reshaped_signal[:, chunk_length // 2].flatten()
            return midpoints
        midpoints = partition_and_calculate_midpoints(list(eeg_block.index))
        accuracy_by_block = [performance_indeces_bin1['accuracy'][0], performance_indeces_bin2['accuracy'][0],
                         performance_indeces_bin3['accuracy'][0], performance_indeces_bin4['accuracy'][0]]
        rt_by_block = [performance_indeces_bin1['mean_rt_target'][0], performance_indeces_bin2['mean_rt_target'][0],
                         performance_indeces_bin3['mean_rt_target'][0], performance_indeces_bin4['mean_rt_target'][0]]
        axs[1].plot([0, 1, 2, 3], accuracy_by_block, color='purple')
        axs[1].set_xticks([0, 1, 2, 3])
        axs[1].set_xticklabels(['', '', '', ''])
        axs[1].set_ylabel('accuracy (%)')
        axs[2].plot([0, 1, 2, 3], rt_by_block, color='black')
        axs[2].set_xticks([0, 1, 2, 3])
        axs[2].set_xticklabels(['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4'])
        axs[2].set_ylabel('reaction time (s)')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        plt.show()

###################################### Correlate blink rate with vigilance ratings #####################################

blink_rate_per_subject = pd.DataFrame(columns=['task', 'subject_id', 'nblinks_overall', 'nblinks_first_half',
                                               'nblinks_second_half', 'mean_blink_rate_overall',
                                               'mean_blink_rate_first_half', 'mean_blink_rate_second_half'])
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        eeg_csv = pd.read_csv(filepath_eeg_csv, index_col=0)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]

        # Set first timestamp to zero
        eeg_block = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index()
        eeg_block['timestamp'] -= eeg_block['timestamp'][0]
        eeg_block.set_index('timestamp', inplace=True)

        # Blink rate overall
        blinks_starts_ends_samples_overall, blinks_timestamps_overall = detect_blinks(eeg_block, thresh_z_blinks, padding_blinks_samples, task, subject_id, visualize=False)
        blinkrate_moving_windows_overall = compute_blinkrate_sliding_window(blinks_timestamps_overall, sliding_window_duration, sliding_window_overlap, block_duration=eeg_block.index[-1])
        nblinks_overall = blinks_starts_ends_samples_overall.shape[0]
        mean_blink_rate_overall = np.mean(blinkrate_moving_windows_overall.blink_rate)

        middle_block = int(eeg_block.shape[0] / 250 / 2)
        eeg_block_first_half = eeg_block.loc[:middle_block]
        eeg_block_second_half = eeg_block.loc[middle_block:]

        # Blink rate first half
        blinks_starts_ends_samples_first_half, blinks_timestamps_first_half = detect_blinks(eeg_block_first_half, thresh_z_blinks, padding_blinks_samples, task, subject_id, visualize=False)
        blinkrate_moving_windows_first_half = compute_blinkrate_sliding_window(blinks_timestamps_first_half, sliding_window_duration, sliding_window_overlap, block_duration=eeg_block.index[-1])
        nblinks_first_half = blinks_starts_ends_samples_first_half.shape[0]
        mean_blink_rate_first_half = np.mean(blinkrate_moving_windows_first_half.blink_rate)

        # Blink rate second half
        blinks_starts_ends_samples_second_half, blinks_timestamps_second_half = detect_blinks(eeg_block_second_half, thresh_z_blinks, padding_blinks_samples, task, subject_id, visualize=False)
        blinkrate_moving_windows_second_half = compute_blinkrate_sliding_window(blinks_timestamps_second_half, sliding_window_duration, sliding_window_overlap, block_duration=eeg_block.index[-1])
        nblinks_second_half = blinks_starts_ends_samples_second_half.shape[0]
        mean_blink_rate_second_half = np.mean(blinkrate_moving_windows_second_half.blink_rate)

        blink_rate_per_subject = pd.concat([blink_rate_per_subject, pd.DataFrame({'task': [task], 'subject_id': [subject_id],
                                                                                  'nblinks_overall': [nblinks_overall],
                                                                                  'nblinks_first_half': [nblinks_first_half],
                                                                                  'nblinks_second_half': [nblinks_second_half],
                                                                                  'mean_blink_rate_overall': [mean_blink_rate_overall],
                                                                                  'mean_blink_rate_first_half': [mean_blink_rate_first_half],
                                                                                  'mean_blink_rate_second_half': [mean_blink_rate_second_half]})])

blink_rate_per_subject = blink_rate_per_subject.merge(questionnaires, on=['task','subject_id'])
blink_rate_per_subject['task_subject'] = blink_rate_per_subject['task'] + '_' + blink_rate_per_subject['subject_id']
# Inverse overall vigilance score since formulated as "difficult to remain vigilant"
blink_rate_per_subject['NASA_vigilance_overall'] = 21 - blink_rate_per_subject['NASA_vigilance_overall']

# Overall
g = sns.lmplot(x="NASA_vigilance_overall", y="mean_blink_rate_overall", hue='subject_id', data=blink_rate_per_subject, palette='Paired', fit_reg=False, height=5, aspect=1.3)
sns.regplot(x="NASA_vigilance_overall", y="mean_blink_rate_overall", data=blink_rate_per_subject, scatter=False, ax=g.axes[0, 0])
corr = scipy.stats.pearsonr(blink_rate_per_subject['NASA_vigilance_overall'], blink_rate_per_subject['mean_blink_rate_overall'])
r = corr[0]
r2 = corr[0]**2
pvalue = '<0.001' if corr[1]<0.001 else str(round(corr[1], 3))
plt.xlabel('vigilance rating', fontsize=12)
plt.ylabel('mean blink rate', fontsize=12)
plt.title('r=' + str(round(r, 2)) + ', p=' + pvalue, fontsize=12)
#plt.tight_layout()
plt.show()

# First half
g = sns.lmplot(x="NASA_vigilance1", y="mean_blink_rate_first_half", hue='task_subject', data=blink_rate_per_subject, palette='Paired', fit_reg=False, height=5, aspect=1.3)
sns.regplot(x="NASA_vigilance1", y="mean_blink_rate_first_half", data=blink_rate_per_subject, scatter=False, ax=g.axes[0, 0])
corr = scipy.stats.pearsonr(blink_rate_per_subject['NASA_vigilance1'], blink_rate_per_subject['mean_blink_rate_first_half'])
r = corr[0]
r2 = corr[0]**2
pvalue = '<0.001' if corr[1]<0.001 else str(round(corr[1], 3))
plt.xlabel('vigilance rating first half', fontsize=12)
plt.ylabel('mean blink rate first half', fontsize=12)
plt.title('r=' + str(round(r, 2)) + ', p=' + pvalue, fontsize=12)
#plt.tight_layout()
plt.show()

# Second half
g = sns.lmplot(x="NASA_vigilance2", y="mean_blink_rate_second_half", hue='task_subject', data=blink_rate_per_subject, palette='Paired', fit_reg=False, height=5, aspect=1.3)
sns.regplot(x="NASA_vigilance2", y="mean_blink_rate_second_half", data=blink_rate_per_subject, scatter=False, ax=g.axes[0, 0])
corr = scipy.stats.pearsonr(blink_rate_per_subject['NASA_vigilance2'], blink_rate_per_subject['mean_blink_rate_second_half'])
r = corr[0]
r2 = corr[0]**2
pvalue = '<0.001' if corr[1]<0.001 else str(round(corr[1], 3))
plt.xlabel('vigilance rating second half', fontsize=12)
plt.ylabel('mean blink rate second half', fontsize=12)
plt.title('r=' + str(round(r, 2)) + ', p=' + pvalue, fontsize=12)
#plt.tight_layout()
plt.show()

# Second minus first half

blink_rate_per_subject['blink_rate_first_minus_second_half'] = blink_rate_per_subject["mean_blink_rate_first_half"] - blink_rate_per_subject["mean_blink_rate_second_half"]
blink_rate_per_subject['vigilance_rating_first_minus_second_half'] = blink_rate_per_subject["NASA_vigilance1"] - blink_rate_per_subject["NASA_vigilance2"]
g = sns.lmplot(x="vigilance_rating_first_minus_second_half", y="blink_rate_first_minus_second_half", hue='task_subject', data=blink_rate_per_subject, palette='Paired', fit_reg=False, height=5, aspect=1.3)
sns.regplot(x="vigilance_rating_first_minus_second_half", y="blink_rate_first_minus_second_half", data=blink_rate_per_subject, scatter=False, ax=g.axes[0, 0])
corr = scipy.stats.pearsonr(blink_rate_per_subject['vigilance_rating_first_minus_second_half'], blink_rate_per_subject['blink_rate_first_minus_second_half'])
r = corr[0]
r2 = corr[0]**2
pvalue = '<0.001' if corr[1]<0.001 else str(round(corr[1], 3))
plt.xlabel('vigilance rating first minus second half', fontsize=12)
plt.ylabel('mean blink rate first minus second half', fontsize=12)
plt.title('r=' + str(round(r, 2)) + ', p=' + pvalue, fontsize=12)
#plt.tight_layout()
plt.show()