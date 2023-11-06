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
                  'line_task_sim': ['02', '03', '04', '18', '21', '25'],
                  'line_task_succ': ['02', '04', '05', '07'],
                  'oddball': ['02', '04', '07']}

### EEG parameters
SAMPLE_RATE = 250
# specify eeg channels
EEG_CHANNELS = ['Pz', 'O1', 'O2', 'F3', 'F4', 'Fp1', 'Fp2']

# Specify frequency bands of interest
FREQ_BANDS = {"delta": [0.5, 4.5],
              "theta": [4.5, 8],
              "alpha": [8, 13],
              "beta": [13, 30]}

codes_triggers = {"START_BLOCK": 10,"END_BLOCK": 11,
                  "START_TRIAL": 20,"END_TRIAL": 21,
                  "BUTTON_PRESS": 7, "ONSET_TARGETS": 30, "OFFSET_TARGETS": 31,
                  "ONSET_NONTARGETS": [40, 50], "OFFSET_NONTARGETS": [41, 51]
                  }

# This is the size for the window to estimate PSD using Welch's method
psd_window_size_sec = 4
psd_window_overlap = 0.5

# Select artefact rejection method: 'threshold' or 'asr' (to implement)
method_artrej = 'threshold'

### Blink parameters

# Select z-threshold for blink detection
thresh_z_blinks = 2.5
padding_blinks = 0.1    # in seconds
padding_blinks_samples = padding_blinks * SAMPLE_RATE

# Parameters of the sliding window to compute blink rate
blinks_sliding_window_duration = 60        # in seconds
blinks_sliding_window_overlap = 0.5

### Load questionnaires
questionnaires = pd.read_csv(raw_dir + '/questionnaires_answers.csv', sep=';', dtype={'Subject ID': str})
#questionnaires = questionnaires.dropna().reset_index(drop=True)
questionnaires["subject_id"] = questionnaires["subject_id"].astype(str).str.zfill(2)

###################################################### Show PSDs #######################################################

psd_window_size_seconds = 4
psd_overlap = 0.5

for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        filepath_eeg_fif = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned_raw.fif'
        eeg_csv = pd.read_csv(filepath_eeg_csv)
        eeg_fif = mne.io.read_raw_fif(filepath_eeg_fif, preload=True)

        window_size_samples = int(psd_window_size_seconds * SAMPLE_RATE)
        samples_overlap = int(window_size_samples * psd_overlap)
        psds, freqs = mne.time_frequency.psd_welch(eeg_fif, picks='eeg', fmin=1, fmax=30, n_fft=window_size_samples, n_overlap=samples_overlap)

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

################################ ALL IN ONE: Power, blink rate, performance and ratings ################################
# Show behavior and questionnaires on top

# Define frequency band of interest
freq_band = 'alpha'

# This is the size of the window that is moved along the signal within which the PSD is computed
size_moving_window_seconds = 20
size_moving_window_samples = size_moving_window_seconds * SAMPLE_RATE
moving_window_overlap = 0.5

# Select channel(s) of interest:
# 'mean_posterior' = Take mean over channels Pz, O1 and O2
# 'mean_Fp1Fp2' = Take mean over channels Fp1 and Fp2
# 'mean_frontal' = Take mean over channels Fp1, Fp2, F3, and F4
channel_of_interest = 'mean_posterior'

if psd_window_size_sec > size_moving_window_seconds:
    raise ValueError('PSD window cannot be larger than moving window')
elif psd_window_size_sec == size_moving_window_seconds:
    psd_window_overlap = 0

for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_log_bad_channels = data_work_path + f'/Artefacts/log_bad_channels_{task}_subject{subject_id}.csv'
        log_bad_channels = pd.read_csv(filepath_log_bad_channels, index_col=0)

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        eeg_csv = pd.read_csv(filepath_eeg_csv, index_col=0)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]

        eeg_block = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index(drop=True)
        # Still need the block with timestamps later to plot the events of interest
        eeg_block_with_timestamps = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index()
        eeg_block_with_timestamps['timestamp'] -= eeg_block_with_timestamps['timestamp'][0]
        eeg_block_with_timestamps.set_index('timestamp', inplace=True)

        results_behavior = read_behavior_files(subject_id, task, raw_dir, block_type='experiment')

        '''
        # Crosscheck the behavior file and trigger information matches
        from collections import Counter
        counter = Counter(results_behavior.type_response)
        print(counter)
        
        print('hits: ' + str(eeg_block.loc[eeg_block.triggers_of_interest=='hit'].shape[0]))
        print('misses: ' + str(eeg_block.loc[eeg_block.triggers_of_interest=='miss'].shape[0]))
        print('false alarmss: ' + str(eeg_block.loc[eeg_block.triggers_of_interest=='false_alarm'].shape[0]))
        '''

        power_moving_windows = pd.DataFrame(columns=['center_window', 'power'])

        sample_start_window = 0
        sample_end_window = size_moving_window_samples
        while sample_end_window <= eeg_block.shape[0]:

            eeg_window = eeg_block.loc[sample_start_window:sample_end_window]
            eeg_window = transform_raw(eeg_window)

            absolute_power_per_channel_band, relative_power_per_channel_band = eeg_band(eeg_window, EEG_CHANNELS, FREQ_BANDS, psd_window_size_sec, psd_window_overlap)
            center_window = int(sample_start_window+(size_moving_window_samples/2)) / 250 # compute center of the window in seconds
            if channel_of_interest == 'mean_posterior':
                power_window = np.nanmean(absolute_power_per_channel_band.loc[['Pz', 'O1', 'O2'], freq_band])
            elif channel_of_interest == 'mean_Fp1Fp2':
                power_window = np.nanmean(absolute_power_per_channel_band.loc[['Fp1', 'Fp2'], freq_band])
            elif channel_of_interest == 'mean_frontal':
                power_window = np.nanmean(absolute_power_per_channel_band.loc[['Fp1', 'Fp2', 'F3', 'F4'], freq_band])
            else:
                power_window = absolute_power_per_channel_band.loc[channel_of_interest, freq_band]
            power_moving_windows = pd.concat([power_moving_windows, pd.DataFrame({'center_window': [center_window],
                                                                                 'power': [power_window]})])

            sample_start_window += int(size_moving_window_samples * moving_window_overlap)
            sample_end_window = sample_start_window + size_moving_window_samples

        # Add triggers of interest
        eeg_block_with_timestamps = insert_triggers_of_interest(eeg_block_with_timestamps, codes_triggers)
        ind_hits = eeg_block_with_timestamps.loc[eeg_block_with_timestamps.triggers_of_interest=='hit'].index
        ind_misses = eeg_block_with_timestamps.loc[eeg_block_with_timestamps.triggers_of_interest=='miss'].index
        ind_false_alarms = eeg_block_with_timestamps.loc[eeg_block_with_timestamps.triggers_of_interest=='false_alarm'].index

        fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 2, 1, 1]}, figsize=(12,8))
        axs.flatten()
        axs[0].plot(power_moving_windows.center_window, power_moving_windows.power)
        # plot line of best fit
        a, b = np.polyfit(list(power_moving_windows.dropna().center_window), list(power_moving_windows.dropna().power), 1)
        axs[0].plot(power_moving_windows.center_window, a*power_moving_windows.center_window+b, color = 'grey', label=f'best fit {freq_band} power')
        axs[0].set_xlabel('center window (seconds)')
        axs[0].set_ylabel(f'{freq_band} power ' + channel_of_interest)
        axs[0].set_title(f'{task}, S{subject_id}, {size_moving_window_seconds} sec. moving window, {moving_window_overlap*100}% overlap')
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
                axs[0].scatter(ind_hit, np.nanmin(power_moving_windows['power']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='green', s=50, label='hits')
                label_added = True
            else:
                axs[0].scatter(ind_hit, np.nanmin(power_moving_windows['power']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='green', s=50)
        label_added = False
        for ind_miss in ind_misses:
            if not label_added:
                axs[0].scatter(ind_miss, np.nanmin(power_moving_windows['power']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='red', s=50, label='misses')
                label_added = True
            else:
                axs[0].scatter(ind_miss, np.nanmin(power_moving_windows['power']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='red', s=50)
        label_added = False
        for ind_false_alarm in ind_false_alarms:
            if not label_added:
                axs[0].scatter(ind_false_alarm, np.nanmin(power_moving_windows['power']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='orange', s=50, label='false alarms')
                label_added = True
            else:
                axs[0].scatter(ind_false_alarm, np.nanmin(power_moving_windows['power']) * (0.9+random.uniform(-0.07, 0.07)), facecolors='none',
                            edgecolors='orange', s=50)
        axs[0].set_ylim([ylim[0]*0.6, ylim[1] * 1.3])
        axs[0].legend(loc='center right')

        # Show blink rate
        if any(eeg_block_with_timestamps['Fp1'].notnull()) and any(eeg_block_with_timestamps['Fp1'].notnull()):
            blinks_starts_ends_samples, blinks_timestamps = detect_blinks(eeg_block_with_timestamps, thresh_z_blinks, padding_blinks_samples, task, subject_id, visualize=False)
            blinkrate_moving_windows = compute_blinkrate_sliding_window(blinks_timestamps, blinks_sliding_window_duration, blinks_sliding_window_overlap, block_duration=eeg_block_with_timestamps.index[-1])
            axs[1].plot(blinkrate_moving_windows.center_window, blinkrate_moving_windows.blink_rate)
            # plot line of best fit
            a, b = np.polyfit(list(blinkrate_moving_windows.dropna().center_window), list(blinkrate_moving_windows.dropna().blink_rate), 1)
            axs[1].plot(blinkrate_moving_windows.center_window, a*blinkrate_moving_windows.center_window+b, color = 'grey', label=f'best fit {freq_band} power')
            axs[1].set_xlabel('center window (seconds)')
            axs[1].set_ylabel('blink rate')
            axs[1].set_title(f'Blink rates from {blinks_sliding_window_duration} sec. moving windows with {blinks_sliding_window_overlap*100}% overlap')

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
        midpoints = partition_and_calculate_midpoints(list(eeg_csv.index))
        accuracy_by_block = [performance_indeces_bin1['accuracy'][0], performance_indeces_bin2['accuracy'][0],
                         performance_indeces_bin3['accuracy'][0], performance_indeces_bin4['accuracy'][0]]
        rt_by_block = [performance_indeces_bin1['mean_rt_target'][0], performance_indeces_bin2['mean_rt_target'][0],
                         performance_indeces_bin3['mean_rt_target'][0], performance_indeces_bin4['mean_rt_target'][0]]
        axs[2].plot([0, 1, 2, 3], accuracy_by_block, color='purple')
        axs[2].set_xticks([0, 1, 2, 3])
        axs[2].set_xticklabels(['', '', '', ''])
        axs[2].set_ylabel('accuracy (%)')
        axs[3].plot([0, 1, 2, 3], rt_by_block, color='black')
        axs[3].set_xticks([0, 1, 2, 3])
        axs[3].set_xticklabels(['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4'])
        axs[3].set_ylabel('reaction time (s)')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        plt.show()

############################# EEG alpha power first vs. second half / beginning vs. end ################################

# Select to take either first vs. second half by 'halves' or integer for first vs. last N minutes
parts = 10

# If true, use relative instead of absolute alpha band power
relative = False

take_log = True

alpha_power_per_half = pd.DataFrame(columns=['task', 'subject_id', 'alpha_first_half_Pz', 'alpha_first_half_O1',
                                             'alpha_first_half_O2', 'alpha_second_half_Pz',
                                             'alpha_second_half_O1', 'alpha_second_half_O2'])
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        eeg_csv = pd.read_csv(filepath_eeg_csv, index_col=0)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]

        eeg_block = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index(drop=True)

        if parts == 'halves':
            middle_block = int(eeg_block.shape[0] / 2)
            eeg_block_first_half = transform_raw(eeg_block.loc[:middle_block])
            eeg_block_second_half = transform_raw(eeg_block.loc[middle_block:])
        else:
            eeg_block_first_half = transform_raw(eeg_block.loc[:(parts*60*SAMPLE_RATE)-1])
            eeg_block_second_half = transform_raw(eeg_block.loc[eeg_block.shape[0]-(parts * 60 * SAMPLE_RATE):])
        absolute_power_per_channel_band_first_half, relative_power_per_channel_band_first_half = eeg_band(eeg_block_first_half, EEG_CHANNELS, FREQ_BANDS, psd_window_size_sec, psd_window_overlap)
        absolute_power_per_channel_band_second_half, relative_power_per_channel_band_second_half = eeg_band(eeg_block_second_half, EEG_CHANNELS, FREQ_BANDS, psd_window_size_sec, psd_window_overlap)
        if relative:
            power_per_channel_band_first_half = relative_power_per_channel_band_first_half
            power_per_channel_band_second_half = relative_power_per_channel_band_second_half
        else:
            power_per_channel_band_first_half = absolute_power_per_channel_band_first_half
            power_per_channel_band_second_half = absolute_power_per_channel_band_second_half
        alpha_first_half_Pz = power_per_channel_band_first_half.loc['Pz', 'alpha']
        alpha_first_half_O1 = power_per_channel_band_first_half.loc['O1', 'alpha']
        alpha_first_half_O2 = power_per_channel_band_first_half.loc['O2', 'alpha']

        alpha_second_half_Pz = power_per_channel_band_second_half.loc['Pz', 'alpha']
        alpha_second_half_O1 = power_per_channel_band_second_half.loc['O1', 'alpha']
        alpha_second_half_O2 = power_per_channel_band_second_half.loc['O2', 'alpha']
        if take_log:
            alpha_first_half_Pz = np.log(alpha_first_half_Pz)
            alpha_first_half_O1 = np.log(alpha_first_half_O1)
            alpha_first_half_O2 = np.log(alpha_first_half_O2)

            alpha_second_half_Pz = np.log(alpha_second_half_Pz)
            alpha_second_half_O1 = np.log(alpha_second_half_O1)
            alpha_second_half_O2 = np.log(alpha_second_half_O2)

        alpha_power_per_half = pd.concat([alpha_power_per_half, pd.DataFrame({'task': [task], 'subject_id': [subject_id],
                                                                              'alpha_first_half_Pz': [alpha_first_half_Pz], 'alpha_first_half_O1': [alpha_first_half_O1],
                                                                              'alpha_first_half_O2': [alpha_first_half_O2], 'alpha_second_half_Pz': [alpha_second_half_Pz],
                                                                              'alpha_second_half_O1': [alpha_second_half_O1], 'alpha_second_half_O2': [alpha_second_half_O2]})])
alpha_power_per_half.reset_index(inplace=True, drop=True)
alpha_power_per_half['task_subject'] = alpha_power_per_half['task'] + '_' + alpha_power_per_half['subject_id']

# Normalize across subjects
for col in alpha_power_per_half.drop(['task', 'subject_id', 'task_subject'], axis=1).columns:
    alpha_power_per_half[col] = (alpha_power_per_half[col]-np.nanmin(alpha_power_per_half[col])) / (np.nanmax(alpha_power_per_half[col]) - np.nanmin(alpha_power_per_half[col]))

# Get maximally distinguishable colors
colors = matplotlib.cm.tab20(range(alpha_power_per_half.shape[0]))

fig, axs = plt.subplots(1, 3)
axs = axs.flatten()
for i in range(alpha_power_per_half.shape[0]):
    axs[0].plot([1, 2], [alpha_power_per_half.loc[i, 'alpha_first_half_Pz'], alpha_power_per_half.loc[i, 'alpha_second_half_Pz']], 'o-', color=colors[i])
if relative:
    axs[0].set_ylabel('relative alpha power', fontsize=18)
else:
    axs[0].set_ylabel('absolute alpha power', fontsize=18)
axs[0].set_xticks([1, 2])
axs[0].set_xlim([0.7, 2.3])
axs[0].set_xticklabels(['first half', 'second half'])
alpha_Pz = alpha_power_per_half[['alpha_first_half_Pz', 'alpha_second_half_Pz']].dropna()
res = stats.ttest_ind(alpha_Pz['alpha_first_half_Pz'], alpha_Pz['alpha_second_half_Pz'])
axs[0].set_title(f'Pz, t={round(res.statistic, 2)}, p={round(res.pvalue, 2)}')
axs[0].tick_params(axis='both', which='major', labelsize=12)
for i in range(alpha_power_per_half.shape[0]):
    axs[1].plot([1, 2], [alpha_power_per_half.loc[i, 'alpha_first_half_O1'], alpha_power_per_half.loc[i, 'alpha_second_half_O1']], 'o-', color=colors[i])
if relative:
    axs[1].set_ylabel('relative alpha power', fontsize=18)
else:
    axs[1].set_ylabel('absolute alpha power', fontsize=18)
axs[1].set_xticks([1, 2])
axs[1].set_xlim([0.7, 2.3])
axs[1].set_xticklabels(['first half', 'second half'])
alpha_O1 = alpha_power_per_half[['alpha_first_half_O1', 'alpha_second_half_O1']].dropna()
res = stats.ttest_ind(alpha_O1['alpha_first_half_O1'], alpha_O1['alpha_second_half_O1'])
axs[1].set_title(f'O1, t={round(res.statistic, 2)}, p={round(res.pvalue, 2)}')
axs[1].tick_params(axis='both', which='major', labelsize=12)
for i in range(alpha_power_per_half.shape[0]):
    axs[2].plot([1, 2], [alpha_power_per_half.loc[i, 'alpha_first_half_O2'], alpha_power_per_half.loc[i, 'alpha_second_half_O2']], 'o-', color=colors[i], label=alpha_power_per_half.loc[i, 'task_subject'])
if relative:
    axs[2].set_ylabel('relative alpha power', fontsize=18)
else:
    axs[2].set_ylabel('absolute alpha power', fontsize=18)
axs[2].set_xticks([1, 2])
axs[2].set_xlim([0.7, 2.3])
axs[2].set_xticklabels(['first half', 'second half'])
alpha_O2 = alpha_power_per_half[['alpha_first_half_O2', 'alpha_second_half_O2']].dropna()
res = stats.ttest_ind(alpha_O2['alpha_first_half_O2'], alpha_O2['alpha_second_half_O2'])
axs[2].set_title(f'O2, t={round(res.statistic, 2)}, p={round(res.pvalue, 2)}')
axs[2].tick_params(axis='both', which='major', labelsize=12)
# plt.legend()
plt.tight_layout()
plt.show()

############################################ Alpha power & vigilance ratings ###########################################

# If true, use relative instead of absolute alpha band power
relative = False

psd_window_size_seconds = 4
psd_window_overlap = 0.5

alpha_power_vigilance_features = pd.DataFrame(columns=['task', 'subject_id', 'alpha_first_half', 'alpha_second_half'])
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        eeg_csv = pd.read_csv(filepath_eeg_csv, index_col=0)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]

        eeg_block = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index(drop=True)

        middle_block = int(eeg_block.shape[0] / 2)
        eeg_block_first_half = transform_raw(eeg_block.loc[:middle_block])
        eeg_block_second_half = transform_raw(eeg_block.loc[middle_block:])
        eeg_block = transform_raw(eeg_block)
        absolute_power_per_channel_band_overall, relative_power_per_channel_band_overall = eeg_band(eeg_block, EEG_CHANNELS,
                                                                                                    FREQ_BANDS, psd_window_size_sec,
                                                                                                    psd_window_overlap)
        absolute_power_per_channel_band_first_half, relative_power_per_channel_band_first_half = eeg_band(eeg_block_first_half, EEG_CHANNELS, FREQ_BANDS, psd_window_size_sec, psd_window_overlap)
        absolute_power_per_channel_band_second_half, relative_power_per_channel_band_second_half = eeg_band(eeg_block_second_half, EEG_CHANNELS, FREQ_BANDS, psd_window_size_sec, psd_window_overlap)
        if relative:
            power_per_channel_band_overall = relative_power_per_channel_band_overall
            power_per_channel_band_first_half = relative_power_per_channel_band_first_half
            power_per_channel_band_second_half = relative_power_per_channel_band_second_half
        else:
            power_per_channel_band_overall = absolute_power_per_channel_band_overall
            power_per_channel_band_first_half = absolute_power_per_channel_band_first_half
            power_per_channel_band_second_half = absolute_power_per_channel_band_second_half
        alpha_overall = np.nanmean(absolute_power_per_channel_band_overall.loc[['Pz', 'O1', 'O2'], 'alpha'])
        alpha_first_half = np.nanmean(absolute_power_per_channel_band_first_half.loc[['Pz', 'O1', 'O2'], 'alpha'])
        alpha_second_half = np.nanmean(absolute_power_per_channel_band_second_half.loc[['Pz', 'O1', 'O2'], 'alpha'])
        vigilance_rating_first_half = questionnaires.loc[(questionnaires.task==task) & (questionnaires.subject_id == subject_id), 'NASA_vigilance1'].values[0]
        vigilance_rating_second_half = questionnaires.loc[(questionnaires.task==task) & (questionnaires.subject_id == subject_id), 'NASA_vigilance2'].values[0]
        sleepiness_rating = questionnaires.loc[
            (questionnaires.task == task) & (questionnaires.subject_id == subject_id), 'sleepiness'].values[0]
        hours_slept = questionnaires.loc[
            (questionnaires.task == task) & (questionnaires.subject_id == subject_id), 'hours_slept'].values[0]

        alpha_power_vigilance_features = pd.concat([alpha_power_vigilance_features,
                                                    pd.DataFrame({'task': [task], 'subject_id': [subject_id],
                                                                  'alpha_overall': [alpha_overall],
                                                                  'alpha_first_half': [alpha_first_half],
                                                                  'alpha_second_half': [alpha_second_half],
                                                                  'vigilance_rating_first_half': [vigilance_rating_first_half],
                                                                  'vigilance_rating_second_half': [vigilance_rating_second_half],
                                                                  'sleepiness': [sleepiness_rating],
                                                                  'hours_slept': [hours_slept]})])

alpha_power_vigilance_features['alpha_second_minus_first_half'] = alpha_power_vigilance_features['alpha_second_half'] - alpha_power_vigilance_features['alpha_first_half']
alpha_power_vigilance_features['vigilance_rating_second_minus_first_half'] = alpha_power_vigilance_features['vigilance_rating_second_half'] - alpha_power_vigilance_features['vigilance_rating_first_half']
alpha_power_vigilance_features['task_subject'] = alpha_power_vigilance_features['task'] + '_' + alpha_power_vigilance_features['subject_id']

## Correlate features of interest

feature1 = "hours_slept"
feature2 = "alpha_overall"
g = sns.lmplot(x=feature1, y=feature2, hue='task_subject', data=alpha_power_vigilance_features, palette='Paired', fit_reg=False, height=5, aspect=1.3)
sns.regplot(x=feature1, y=feature2, data=alpha_power_vigilance_features, scatter=False, ax=g.axes[0, 0])
corr = scipy.stats.pearsonr(alpha_power_vigilance_features[feature1], alpha_power_vigilance_features[feature2])
r = corr[0]
r2 = corr[0]**2
pvalue = '<0.001' if corr[1]<0.001 else str(round(corr[1], 3))
plt.xlabel(feature1, fontsize=12)
plt.ylabel(feature2, fontsize=12)
plt.title('r=' + str(round(r, 2)) + ', p=' + pvalue, fontsize=12)
#plt.tight_layout()
plt.show()

#################################### Analyze EEG alpha power and performance by bin ####################################

alpha_perf_per_bin = pd.DataFrame(columns = ['task','subject','bin','alpha','ncorrect_rejections', 'nhits',
                                             'nfalse_alarms', 'nmisses', 'hit_rate', 'accuracy', 'dprime', 'criterion',
                                             'mean_rt_target', 'mean_rt_overall', 'ies'])
for task in tasks_subjects.keys():

    print(f'##### {task}...')

    for subject_id in tasks_subjects[task]:

        print(f'### S{subject_id}...')

        results_behavior = read_behavior_files(subject_id, task, raw_dir, block_type='experiment')

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        eeg_csv = pd.read_csv(filepath_eeg_csv, index_col=0)

        ind_starts_blocks = list(eeg_csv.loc[eeg_csv.trigger == codes_triggers["START_BLOCK"]].index)
        ind_ends_blocks = list(eeg_csv.loc[eeg_csv.trigger == codes_triggers["END_BLOCK"]].index)

        # Partition eeg into 4 bins: For oddball task, we have 4 blocks already. For others, partition trials into 4
        # balanced bins
        if task == 'oddball':

            performance_indeces_bin1 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==1])
            performance_indeces_bin2 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==2])
            performance_indeces_bin3 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==3])
            performance_indeces_bin4 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==4])

            eeg_bin1 = transform_raw(eeg_csv.loc[ind_starts_blocks[1]:ind_ends_blocks[1]])
            eeg_bin2 = transform_raw(eeg_csv.loc[ind_starts_blocks[2]:ind_ends_blocks[2]])
            eeg_bin3 = transform_raw(eeg_csv.loc[ind_starts_blocks[3]:ind_ends_blocks[3]])
            eeg_bin4 = transform_raw(eeg_csv.loc[ind_starts_blocks[4]:ind_ends_blocks[4]])

        else:
            ntrials = results_behavior.shape[0]
            ntrials_per_bin = distribute_items(ntrials, 4)

            performance_indeces_bin1 = compute_performance_indeces(results_behavior.loc[:ntrials_per_bin[0]-1])
            performance_indeces_bin2 = compute_performance_indeces(results_behavior.loc[ntrials_per_bin[0]:ntrials_per_bin[0]+(ntrials_per_bin[1]-1)])
            performance_indeces_bin3 = compute_performance_indeces(results_behavior.loc[(ntrials_per_bin[0]+ntrials_per_bin[1]):(ntrials_per_bin[0]+ntrials_per_bin[1])+(ntrials_per_bin[2]-1)])
            performance_indeces_bin4 = compute_performance_indeces(results_behavior.loc[(ntrials_per_bin[0]+ntrials_per_bin[1]+ntrials_per_bin[2]):])

            # extract eeg during actual experiment
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
            eeg_experiment = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index(drop=True)

            ind_starts_trials = list(eeg_experiment.loc[eeg_experiment.trigger == codes_triggers["START_TRIAL"]].index)
            ind_ends_trials = list(eeg_experiment.loc[eeg_experiment.trigger == codes_triggers["END_TRIAL"]].index)
            if len(ind_ends_trials) == 0:   # In line task, end of trial triggers were not sent, note them from start trial triggers
                warnings.warn('Found no triggers for end of trials. Reconstructing from start trial triggers')
                ind_ends_trials = [i-1 for i in ind_starts_trials[1:]] + [eeg_experiment.loc[eeg_experiment.trigger == codes_triggers["END_BLOCK"]].index[-1]]

            if len(ind_starts_trials) != ntrials:
                raise ValueError('number of trial triggers does not match number of trials according to behavior file')

            trials_bin1 = [0, ntrials_per_bin[0]-1]
            trials_bin2 = [ntrials_per_bin[0], ntrials_per_bin[0]+ntrials_per_bin[1]-1]
            trials_bin3 = [ntrials_per_bin[0]+ntrials_per_bin[1], ntrials_per_bin[0]+ntrials_per_bin[1]+ntrials_per_bin[2]-1]
            trials_bin4 = [ntrials_per_bin[0]+ntrials_per_bin[1]+ntrials_per_bin[2], ntrials_per_bin[0]+ntrials_per_bin[1]+ntrials_per_bin[2]+ntrials_per_bin[3]-1]

            eeg_bin1 = transform_raw(eeg_experiment.loc[ind_starts_trials[0]:ind_ends_trials[trials_bin1[1]]])
            eeg_bin2 = transform_raw(eeg_experiment.loc[ind_starts_trials[trials_bin2[0]]:ind_ends_trials[trials_bin2[1]]])
            eeg_bin3 = transform_raw(eeg_experiment.loc[ind_starts_trials[trials_bin3[0]]:ind_ends_trials[trials_bin3[1]]])
            eeg_bin4 = transform_raw(eeg_experiment.loc[ind_starts_trials[trials_bin4[0]]:ind_ends_trials[trials_bin4[1]]])

        performance_indeces_bin1 = compute_performance_indeces(results_behavior.loc[:ntrials_per_bin[0]-1])
        performance_indeces_bin2 = compute_performance_indeces(results_behavior.loc[ntrials_per_bin[0]:ntrials_per_bin[0]+(ntrials_per_bin[1]-1)])
        performance_indeces_bin3 = compute_performance_indeces(results_behavior.loc[(ntrials_per_bin[0]+ntrials_per_bin[1]):(ntrials_per_bin[0]+ntrials_per_bin[1])+(ntrials_per_bin[2]-1)])
        performance_indeces_bin4 = compute_performance_indeces(results_behavior.loc[(ntrials_per_bin[0]+ntrials_per_bin[1]+ntrials_per_bin[2]):])

        absolute_power_per_channel_band_bin1, _ = eeg_band(eeg_bin1, EEG_CHANNELS, FREQ_BANDS, window_size_sec, window_overlap)
        absolute_power_per_channel_band_bin2, _ = eeg_band(eeg_bin2, EEG_CHANNELS, FREQ_BANDS, window_size_sec, window_overlap)
        absolute_power_per_channel_band_bin3, _ = eeg_band(eeg_bin3, EEG_CHANNELS, FREQ_BANDS, window_size_sec, window_overlap)
        absolute_power_per_channel_band_bin4, _ = eeg_band(eeg_bin4, EEG_CHANNELS, FREQ_BANDS, window_size_sec, window_overlap)
        alpha_bin1 = np.nanmean(absolute_power_per_channel_band_bin1.loc[['Pz', 'O1', 'O2'], 'alpha'])
        alpha_bin2 = np.nanmean(absolute_power_per_channel_band_bin2.loc[['Pz', 'O1', 'O2'], 'alpha'])
        alpha_bin3 = np.nanmean(absolute_power_per_channel_band_bin3.loc[['Pz', 'O1', 'O2'], 'alpha'])
        alpha_bin4 = np.nanmean(absolute_power_per_channel_band_bin4.loc[['Pz', 'O1', 'O2'], 'alpha'])

        results_bin1 = pd.concat([performance_indeces_bin1, pd.DataFrame({'alpha': [alpha_bin1]})], axis=1)
        alpha_perf_per_bin = pd.concat([alpha_perf_per_bin, pd.concat([pd.DataFrame({'task':[task],'subject':[subject_id],'bin':[1]}), results_bin1], axis=1)])

### Correlate Sleepiness questionnaire scores with overall alpha power and in beginning (first 10 minutes)

psd_window_size_seconds = 4
psd_overlap = 0.5

alpha_power_per_subject = pd.DataFrame(columns=['task', 'subject_id', 'alpha_power_overall', 'alpha_power_beginning'])
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        print(f'{task}, S{subject_id}...')

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        filepath_eeg_fif = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned_raw.fif'
        eeg_csv = pd.read_csv(filepath_eeg_csv)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger==codes_triggers["END_BLOCK"]].index)[-1]

        eeg_block = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index(drop=True)
        eeg_block.timestamp = eeg_block.timestamp - eeg_block.timestamp[0]
        eeg_block.set_index('timestamp', inplace=True)

        absolute_power_per_channel_band, relative_power_per_channel_band = eeg_band(transform_raw(eeg_block), EEG_CHANNELS,
                                                                                    FREQ_BANDS, psd_window_size_seconds, psd_overlap)
        alpha_overall = np.nanmean(absolute_power_per_channel_band.loc[['Pz', 'O1', 'O2'], 'alpha'])

        # Extract the first 10 minutes
        eeg_beginning = eeg_block.loc[eeg_block.index <= (10 * 60)]
        absolute_power_per_channel_band, relative_power_per_channel_band = eeg_band(transform_raw(eeg_block), EEG_CHANNELS,
                                                                                    FREQ_BANDS, psd_window_size_seconds, psd_overlap)
        alpha_beginning = np.nanmean(absolute_power_per_channel_band.loc[['Pz', 'O1', 'O2'], 'alpha'])

        alpha_power_per_subject = pd.concat([alpha_power_per_subject, pd.DataFrame({'task': [task], 'subject_id': [subject_id],
                                                                                    'alpha_overall': [alpha_overall], 'alpha_beginning': [alpha_beginning]})])

alpha_power_questionnaires_per_subject = alpha_power_per_subject.merge(questionnaires, on=['task','subject_id'])
alpha_power_questionnaires_per_subject['drop_vigilance'] = alpha_power_questionnaires_per_subject['NASA_vigilance2'] - alpha_power_questionnaires_per_subject['NASA_vigilance1']

############################## Show PSDs for beginning vs. end of experiment, by channel ###############################

# Duration of the portion to extract from the beginning and end in minutes
duration_portion_beginning_end = 10

psd_window_size_seconds = 4
psd_overlap = 0.5
window_size_samples = int(psd_window_size_seconds * SAMPLE_RATE)
samples_overlap = int(window_size_samples / 2)
fmin = 1
fmax = 30

for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch_cleaned_art_{method_artrej}/{task}_S{subject_id}/{task}_S{subject_id}_eeg_cleaned.csv'
        eeg_csv = pd.read_csv(filepath_eeg_csv, index_col=0)

        # Extract data from the experimental block only (corresponds to last block triggers)
        # Oddball task has one training and four experimental blocks
        if task == 'oddball':
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger == codes_triggers["START_BLOCK"]].index)[1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger == codes_triggers["END_BLOCK"]].index)[-1]
        else:
            ind_start_exp = list(eeg_csv.loc[eeg_csv.trigger == codes_triggers["START_BLOCK"]].index)[-1]
            ind_end_exp = list(eeg_csv.loc[eeg_csv.trigger == codes_triggers["END_BLOCK"]].index)[-1]

        eeg_block = eeg_csv.loc[ind_start_exp:ind_end_exp].reset_index()
        eeg_block['timestamp'] -= eeg_block['timestamp'][0]
        eeg_block.set_index('timestamp', inplace=True)

        eeg_beginning = eeg_block.loc[eeg_block.index<=(duration_portion_beginning_end*60)]
        eeg_end = eeg_block.loc[eeg_block.index>=(eeg_block.index[-1]-(duration_portion_beginning_end*60))]

        psds_beginning, freqs = mne.time_frequency.psd_welch(transform_raw(eeg_beginning), picks='eeg',
                                                             fmin=fmin, fmax=fmax, n_fft=window_size_samples,
                                                             n_overlap=samples_overlap)
        psds_end, freqs = mne.time_frequency.psd_welch(transform_raw(eeg_end), picks='eeg',
                                                       fmin=fmin, fmax=fmax, n_fft=window_size_samples,
                                                       n_overlap=samples_overlap)

        fig, axs = plt.subplots(4, 2)
        axs = axs.flatten()
        for ichannel, channel in enumerate(EEG_CHANNELS):
            if ichannel==(len(EEG_CHANNELS)-1):
                axs[ichannel].plot(freqs, np.log(psds_beginning[ichannel]), color='green', label=f'beginning experiment (first {duration_portion_beginning_end} minutes)')
                axs[ichannel].plot(freqs, np.log(psds_end[ichannel]), color='red', label=f'end experiment (last {duration_portion_beginning_end} minutes)')
            else:
                axs[ichannel].plot(freqs, np.log(psds_beginning[ichannel]), color='green')
                axs[ichannel].plot(freqs, np.log(psds_end[ichannel]), color='red')
            axs[ichannel].set_xlabel('frequency (Hz)')
            axs[ichannel].set_ylabel('log power')
            axs[ichannel].set_title(channel)
        plt.tight_layout()
        plt.suptitle(task + ', S' + subject_id, fontsize = 15)
        fig.delaxes(axs[-1])
        fig.legend(loc='lower right')
        plt.show()


################################################# Event-locked analysis ################################################

# Select epoch duration in seconds
epoch_duration = 2

# Define frequency bands of interest
freq_band = [8, 13]

# Select channel(s) of interest: Pz, O1, O2, 'O1/O2', 'avg_posterior', 'Fp1/Fp2', or 'frontal' (Fp1, Fp2, F3, F4)
channel = 'avg_posterior'

# Amplitude artifacts parameters
cleanHighAmpArtifacts = True
HIGH_AMP_CRITERION = 500  # 0.5 mV
FLAT_CHANNEL_CRITERION = 1  # 1 µV
if cleanHighAmpArtifacts:
    reject_criteria = dict(eeg=HIGH_AMP_CRITERION)
    flat_criteria = dict(eeg=FLAT_CHANNEL_CRITERION)
else:
    reject_criteria = {}
    flat_criteria = {}

# Assign new codes for triggers of interest that will be epoched to
codes_triggers_of_interest = {'correct_rejection': 1, 'hit': 2, 'miss': 3, 'false_alarm': 4}
eeg_epochs_all = []
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        filepath_eeg_csv = data_work_path + f'EEG_raw_notch/S' + subject_id + '/' + 'S' + subject_id + '_eeg_notch.csv'
        filepath_eeg_fif = data_work_path + f'EEG_raw_notch/S' + subject_id + '/' + 'S' + subject_id + '_eeg_notch_raw.fif'
        eeg_csv = pd.read_csv(filepath_eeg_csv)
        eeg_fif = mne.io.read_raw_fif(filepath_eeg_fif, preload=True)

        # Add triggers of interest (hits, misses, false alarms etc.)
        eeg_csv = insert_triggers_of_interest(eeg_csv, codes_triggers)

        # Note events
        events = []
        for idx_event in eeg_csv.triggers_of_interest.dropna().index:
            trigger_code = codes_triggers_of_interest[eeg_csv.loc[idx_event, 'triggers_of_interest']]
            if len(events) == 0:
                events = np.array([idx_event, 0, trigger_code])
            else:
                events = np.vstack((events, np.array([idx_event, 0, trigger_code])))

        eeg_epoched = mne.Epochs(eeg_fif, events, event_id=codes_triggers_of_interest, tmin=-epoch_duration,
                                 tmax=0, baseline=None, preload=True,
                                 verbose=False, on_missing='warn', reject=reject_criteria, flat=flat_criteria)

        if len(eeg_epochs_all) == 0:
            eeg_epochs_all = [eeg_epoched]
        else:
            eeg_epochs_all = eeg_epochs_all + [eeg_epoched]

eeg_epochs_all = mne.concatenate_epochs(eeg_epochs_all)

### Compute average power per response type
psd_window_size_seconds = 2
window_size_samples = int(psd_window_size_seconds * SAMPLE_RATE)
samples_overlap = 0

## Correct rejections
psds_epochs_correct_rejections, freqs = mne.time_frequency.psd_welch(eeg_epochs_all['correct_rejection'], picks='eeg',
                                           fmin=freq_band[0], fmax=freq_band[1], n_fft=window_size_samples,
                                           n_overlap=samples_overlap)
if channel=='Pz':
    psds_epochs_correct_rejections = psds_epochs_correct_rejections[:, 0, :]
elif channel=='O1':
    psds_epochs_correct_rejections = psds_epochs_correct_rejections[:, 1, :]
elif channel=='O1':
    psds_epochs_correct_rejections = psds_epochs_correct_rejections[:, 2, :]
elif channel=='O1/O2':
    psds_epochs_correct_rejections = np.nanmean(psds_epochs_correct_rejections[:, 1:3, :], axis=1)
elif channel=='avg_posterior':
    psds_epochs_correct_rejections = np.nanmean(psds_epochs_correct_rejections[:, 0:3, :], axis=1)
elif channel=='Fp1/Fp2':
    psds_epochs_correct_rejections = np.nanmean(psds_epochs_correct_rejections[:, 5:7, :], axis=1)
elif channel=='frontal':
    psds_epochs_correct_rejections = np.nanmean(psds_epochs_correct_rejections[:, 3:7, :], axis=1)
mean_power_per_epoch_correct_rejections = np.mean(psds_epochs_correct_rejections, axis=1)
avg_power_epochs_correct_rejections = psds_epochs_correct_rejections.mean(axis=0)
avg_psd_correct_rejections = psds_epochs_correct_rejections.mean(axis=0)
stde_psd_correct_rejections = psds_epochs_correct_rejections.std(axis=0) / psds_epochs_correct_rejections.shape[0]

## Hits
psds_epochs_hits, freqs = mne.time_frequency.psd_welch(eeg_epochs_all['hit'], picks='eeg',
                                           fmin=freq_band[0], fmax=freq_band[1], n_fft=window_size_samples,
                                           n_overlap=samples_overlap)
if channel=='Pz':
    psds_epochs_hits = psds_epochs_hits[:, 0, :]
elif channel=='O1':
    psds_epochs_hits = psds_epochs_hits[:, 1, :]
elif channel=='O1':
    psds_epochs_hits = psds_epochs_hits[:, 2, :]
elif channel=='O1/O2':
    psds_epochs_hits = np.nanmean(psds_epochs_hits[:, 1:3, :], axis=1)
elif channel=='avg_posterior':
    psds_epochs_hits = np.nanmean(psds_epochs_hits[:, 0:3, :], axis=1)
elif channel=='Fp1/Fp2':
    psds_epochs_hits = np.nanmean(psds_epochs_hits[:, 5:7, :], axis=1)
elif channel=='frontal':
    psds_epochs_hits = np.nanmean(psds_epochs_hits[:, 3:7, :], axis=1)
mean_power_per_epoch_hits = np.mean(psds_epochs_hits, axis=1)
avg_power_epochs_hits = psds_epochs_hits.mean(axis=0)
avg_psd_hits = psds_epochs_hits.mean(axis=0)
stde_psd_hits = psds_epochs_hits.std(axis=0) / psds_epochs_hits.shape[0]

## Misses
psds_epochs_misses, freqs = mne.time_frequency.psd_welch(eeg_epochs_all['miss'], picks='eeg',
                                           fmin=freq_band[0], fmax=freq_band[1], n_fft=window_size_samples,
                                           n_overlap=samples_overlap)
if channel=='Pz':
    psds_epochs_misses = psds_epochs_misses[:, 0, :]
elif channel=='O1':
    psds_epochs_misses = psds_epochs_misses[:, 1, :]
elif channel=='O1':
    psds_epochs_misses = psds_epochs_misses[:, 2, :]
elif channel=='O1/O2':
    psds_epochs_misses = psds_epochs_misses[:, 1:3, :].mean(axis=1)
elif channel=='avg_posterior':
    psds_epochs_misses = psds_epochs_misses[:, 0:3, :].mean(axis=1)
elif channel=='Fp1/Fp2':
    psds_epochs_misses = psds_epochs_misses[:, 5:7, :].mean(axis=1)
elif channel=='frontal':
    psds_epochs_misses = psds_epochs_misses[:, 3:7, :].mean(axis=1)
mean_power_per_epoch_misses = np.mean(psds_epochs_misses, axis=1)
avg_power_epochs_misses = psds_epochs_misses.mean(axis=0)
avg_psd_misses = psds_epochs_misses.mean(axis=0)
stde_psd_misses = psds_epochs_misses.std(axis=0) / psds_epochs_misses.shape[0]

## False alarms
psds_epochs_false_alarms, freqs = mne.time_frequency.psd_welch(eeg_epochs_all['false_alarm'], picks='eeg',
                                           fmin=freq_band[0], fmax=freq_band[1], n_fft=window_size_samples,
                                           n_overlap=samples_overlap)
if channel=='Pz':
    psds_epochs_false_alarms = psds_epochs_false_alarms[:, 0, :]
elif channel=='O1':
    psds_epochs_false_alarms = psds_epochs_false_alarms[:, 1, :]
elif channel=='O1':
    psds_epochs_false_alarms = psds_epochs_false_alarms[:, 2, :]
elif channel=='O1/O2':
    psds_epochs_false_alarms = psds_epochs_false_alarms[:, 1:3, :].mean(axis=1)
elif channel=='avg_posterior':
    psds_epochs_false_alarms = psds_epochs_false_alarms[:, 0:3, :].mean(axis=1)
elif channel=='Fp1/Fp2':
    psds_epochs_false_alarms = psds_epochs_false_alarms[:, 5:7, :].mean(axis=1)
elif channel=='frontal':
    psds_epochs_false_alarms = psds_epochs_false_alarms[:, 3:7, :].mean(axis=1)
mean_power_per_epoch_false_alarms = np.mean(psds_epochs_false_alarms, axis=1)
avg_power_epochs_false_alarms = psds_epochs_false_alarms.mean(axis=0)
avg_psd_false_alarms = psds_epochs_false_alarms.mean(axis=0)
stde_psd_false_alarms = psds_epochs_false_alarms.std(axis=0) / psds_epochs_false_alarms.shape[0]

plt.figure()
plt.plot(freqs, avg_psd_correct_rejections, color = 'lime', label='correct rejections')
plt.fill_between(freqs, avg_psd_correct_rejections-stde_psd_correct_rejections, avg_psd_correct_rejections+stde_psd_correct_rejections, color='lime', alpha=0.5)
plt.plot(freqs, avg_psd_hits, color = 'darkgreen', label='hits')
plt.plot(freqs, avg_psd_misses, color = 'red', label='misses')
plt.plot(freqs, avg_psd_false_alarms, color = 'orange', label='false alarms')
plt.xlabel('frequency (Hz)')
plt.ylabel('power')
plt.title(f'Average power {epoch_duration} seconds before stimulus onset, channel {channel}')
plt.legend()
plt.show()


t, p = scipy.stats.ttest_ind(mean_power_per_epoch_correct_rejections,
                             mean_power_per_epoch_misses,
                             equal_var=False)
t, p = scipy.stats.ttest_ind(mean_power_per_epoch_hits,
                             mean_power_per_epoch_false_alarms,
                             equal_var=False)