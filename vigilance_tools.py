import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import simps
from collections import Counter
import glob
import warnings
warnings.filterwarnings("ignore")

import mne
from mne import create_info, concatenate_raws
from mne.channels import make_standard_montage
from mne.io import RawArray

# Transforming from csv to raw
def transform_raw(data_csv, sample_rate=250):
    if 'timestamp' in data_csv.columns:
        data_csv.set_index('timestamp', inplace=True)

    if any(data_csv.columns.str.contains('^Unnamed')):
        data_csv = data_csv.loc[:, ~data_csv.columns.str.contains('^Unnamed')]

    # transform to raw format
    d = data_csv.copy()
    ch_names = list(d.columns.values)
    # ch_names[ch_names.index('ECG')] = 'Fz'
    # ch_types = ['eeg'] * 3 + ['ecg'] + ['eeg'] * 4 + ['stim']
    # ECG has to be treated like EEG channel, else notch filter doesn't work
    ch_types = ['eeg'] * (len(data_csv.columns) - 1) + ['stim']
    dValues = d.values.T
    dValues[:len(ch_names) - 1] *= 1e-6  # convert from Volts into µVolts

    infor = create_info(ch_names, sample_rate, ch_types)
    data_raw = RawArray(dValues, info=infor, verbose=False)
    data_raw.set_montage(make_standard_montage('standard_1020'))

    return data_raw

def eeg_band(data, eeg_channels, freq_bands, window_size_sec, window_overlap, sample_rate=250):
    '''Computes band power'''
    channel_names = data.copy().info.ch_names
    channel_names.remove('trigger')
    if channel_names != eeg_channels:
        raise ValueError('EEG channels in data do not correspond to eeg_channels as defined')

    absolute_power_per_channel_band = pd.DataFrame(columns = ['channel'] + list(freq_bands.keys()))
    absolute_power_per_channel_band['channel'] = eeg_channels
    absolute_power_per_channel_band.set_index('channel', inplace=True)
    normalized_power_per_channel_band = absolute_power_per_channel_band.copy()

    window_size_samples = int(window_size_sec * sample_rate)
    samples_overlap = int(window_size_samples * window_overlap)

    psds, freqs = mne.time_frequency.psd_welch(data, picks='eeg', fmin=0.5, fmax=30,
                                               n_fft=window_size_samples, n_overlap=samples_overlap)
    # Absolute bandpower
    for freq_band in freq_bands.keys():
        # Absolute power = area under the curve
        absolute_power_per_channel_band[freq_band] = \
            simps(psds[:, (freqs >= freq_bands[freq_band][0]) & (freqs < freq_bands[freq_band][1])], dx=sample_rate)

    # Normalized bandpower = absolute power divided by total power
    for freq_band in freq_bands.keys():
        for channel in channel_names:
            normalized_power_per_channel_band.loc[channel, freq_band] = \
                absolute_power_per_channel_band.loc[channel, freq_band] / absolute_power_per_channel_band.loc[channel].sum()

    return absolute_power_per_channel_band, normalized_power_per_channel_band

def distribute_items(nitems, groups):
    '''Distributes items over groups as balanced as possible'''
    base_items_per_group = nitems // groups
    remaining_items = nitems % groups
    distribution = [base_items_per_group] * groups
    for i in range(remaining_items):
        distribution[i] += 1

    return distribution

def read_behavior_files(subject_id, task, raw_dir, block_type='experiment'):
    """read_behavior_files: Reads in the behavioral results file for a given subject and merges them into one"""

    behavior_dir = raw_dir + '\\' + task + '\\' + 'Behavior' + '\\' + subject_id + '\\'
    if not os.path.isdir(behavior_dir):
        raise ValueError('Path does not exist: ' + behavior_dir)

    # Get a list of CSV file paths in the directory
    csv_files = glob.glob(behavior_dir + '*.csv')
    csv_files = [file for file in csv_files if ('results_' + subject_id) in file]
    # Make sure training comes first, then practice, then other files
    for file in csv_files:
        if 'practice' in file:
            csv_files.remove(file)
            csv_files = [file] + csv_files
    for file in csv_files:
        if 'training' in file:
            csv_files.remove(file)
            csv_files = [file] + csv_files
    # Initialize an empty list to store the individual DataFrames
    results_behavior = pd.DataFrame()
    # Iterate over each CSV file and load it as a DataFrame
    for file in csv_files:
        results_behavior_block = pd.read_csv(file, index_col=None)
        results_behavior = pd.concat([results_behavior, results_behavior_block])
    results_behavior.reset_index(inplace=True, drop=True)

    results_behavior = results_behavior.loc[results_behavior.block_type == 'experiment']
    results_behavior.reset_index(inplace=True, drop=True)

    return results_behavior

def calculate_dprime(nhits, nfalse_alarms, n_target_trials, n_nontarget_trials):
    """
    Calculate dprime using hit rate, false alarm rate, number of signal trials, and number of noise trials.

    Parameters:
        nhits (int): Number of correct responses (hits).
        nfalse_alarms (int): Number of incorrect responses (false alarms).
        n_target_trials (int): Total number of signal (target) trials.
        n_nontarget_trials (int): Total number of noise (non-target) trials.

    Returns:
        float: The calculated dprime value.
    """
    # Ensure input values are valid and within bounds
    if nhits < 0 or nfalse_alarms < 0 or n_target_trials <= 0 or n_nontarget_trials <= 0:
        raise ValueError("Input values must be non-negative, and n_target_trials and n_nontarget_trials must be greater than 0.")

    # Ensure hit rate and false alarm rate are within [0, 1]
    hit_rate = min(1.0, nhits / n_target_trials)
    false_alarm_rate = min(1.0, nfalse_alarms / n_nontarget_trials)

    # Calculate dprime using the formula: d' = Z(hit_rate) - Z(false_alarm_rate)
    dprime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)

    if math.isinf(dprime): dprime = np.nan;

    return dprime

def calculate_criterion(nhits, nfalse_alarms, n_target_trials, n_nontarget_trials):
    """
    Calculate criterion using hit rate and false alarm rate.

    Parameters:
        nhits (int): Number of correct responses (nhits).
        nfalse_alarms (int): Number of incorrect responses (false alarms).
        n_target_trials (int): Total number of signal (target) trials.
        n_nontarget_trials (int): Total number of noise (non-target) trials.

    Returns:
        float: The calculated criterion value.
    """
    # Ensure input values are valid and within bounds
    if nhits < 0 or nfalse_alarms < 0:
        raise ValueError("Input values must be non-negative.")

    hit_rate = nhits / n_target_trials
    if hit_rate == 1:
        warnings.warn('Cannot compute D prime because hit rate is 1')
    elif hit_rate == 0:
        warnings.warn('Cannot compute D prime because hit rate is 0')

    false_alarm_rate = min(1.0, nfalse_alarms / n_nontarget_trials)
    if false_alarm_rate == 1:
        warnings.warn('Cannot compute D prime because false alarm rate is 1')
    elif false_alarm_rate == 0:
        warnings.warn('Cannot compute D prime because false alarm rate is 0')

    # Calculate the criterion using the formula: C = -0.5 * (Z(hit_rate) + Z(false_alarm_rate))
    from scipy.stats import norm
    criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(false_alarm_rate))

    return criterion

def compute_performance_indeces(results_behavior):
    """
    Calculate all the performance indeces for a given subject.

    Parameters:
        results_behavior (pandas DataFrame): The behavioral results file for a given subject

    Returns:
        pandas DataFrame: Performance indeces
    """
    counter = Counter(results_behavior.type_response)
    ncorrect_rejections = counter['correct_rejection']
    nhits = counter['hit']
    nfalse_alarms = counter['false_alarm']
    nmisses = counter['miss']
    ntrials = (ncorrect_rejections + nhits + nfalse_alarms + nmisses)
    n_target_trials = nhits + nmisses
    n_nontarget_trials = ncorrect_rejections + nfalse_alarms
    hit_rate = nhits / n_target_trials
    accuracy = (nhits + ncorrect_rejections) / ntrials

    dprime = calculate_dprime(nhits, nfalse_alarms, n_target_trials, n_nontarget_trials)
    criterion = calculate_criterion(nhits, nfalse_alarms, n_target_trials, n_nontarget_trials)

    mean_rt_target = np.nanmean(results_behavior.loc[results_behavior.type_response=='hit', 'rt'])
    mean_rt_overall = np.nanmean(results_behavior.rt)

    # inverse efficiency score
    ies = mean_rt_target / accuracy

    performance_indeces = pd.DataFrame({'ncorrect_rejections': [ncorrect_rejections], 'nhits': [nhits],
                                        'nfalse_alarms': [nfalse_alarms], 'nmisses': [nmisses],
                                        'hit_rate': [hit_rate], 'accuracy': [accuracy], 'dprime': [dprime],
                                        'criterion': [criterion], 'mean_rt_target': [mean_rt_target],
                                        'mean_rt_overall': [mean_rt_overall], 'ies': [ies]})

    return performance_indeces

def compute_performance_per_bin(results_behavior, task):
    '''Computes performance (accuracy, dprime, inverse efficiency score, ... over four bins'''
    # For oddball task, there are 4 blocks already
    # Other tasks need to be divided into bins based on trials
    if task == 'oddball':
        performance_indeces_bin1 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==1])
        performance_indeces_bin2 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==2])
        performance_indeces_bin3 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==3])
        performance_indeces_bin4 = compute_performance_indeces(results_behavior.loc[results_behavior.block_number==4])

    else:
        ntrials = results_behavior.shape[0]
        ntrials_per_bin = distribute_items(ntrials, 4)
        performance_indeces_bin1 = compute_performance_indeces(results_behavior.loc[:ntrials_per_bin[0]-1])
        performance_indeces_bin2 = compute_performance_indeces(results_behavior.loc[ntrials_per_bin[0]:ntrials_per_bin[0]+(ntrials_per_bin[1]-1)])
        performance_indeces_bin3 = compute_performance_indeces(results_behavior.loc[(ntrials_per_bin[0]+ntrials_per_bin[1]):(ntrials_per_bin[0]+ntrials_per_bin[1])+(ntrials_per_bin[2]-1)])
        performance_indeces_bin4 = compute_performance_indeces(results_behavior.loc[(ntrials_per_bin[0]+ntrials_per_bin[1]+ntrials_per_bin[2]):])

    return performance_indeces_bin1, performance_indeces_bin2, performance_indeces_bin3, performance_indeces_bin4

def insert_triggers_of_interest(eeg_csv, codes_triggers):

    eeg_with_new_triggers = eeg_csv.copy()
    eeg_with_new_triggers['triggers_of_interest'] = np.nan

    ind_button_presses = eeg_csv.loc[eeg_csv.trigger==codes_triggers["BUTTON_PRESS"]].index
    ind_onsets_targets = eeg_csv.loc[eeg_csv.trigger==codes_triggers["ONSET_TARGETS"]].index
    ind_onsets_nontargets = eeg_csv.loc[eeg_csv.trigger.isin(codes_triggers["ONSET_NONTARGETS"])].index
    ind_stim_onsets = sorted(list(ind_onsets_targets) + list(ind_onsets_nontargets))

    # Mark correct rejections
    for inontarget in range(len(ind_onsets_nontargets)):
        onset_current_stim = ind_onsets_nontargets[inontarget]
        if any(ind_stim_onsets > onset_current_stim):
            onset_next_stim = ind_stim_onsets[np.min([i for i in range(len(ind_stim_onsets)) if ind_stim_onsets[i] > onset_current_stim])]
            # Check if there is a button press between the current target and the next stimulus
            button_press_in_between = [onset_current_stim < x < onset_next_stim for x in ind_button_presses]
            if not any(button_press_in_between):
                eeg_with_new_triggers.loc[onset_current_stim, 'triggers_of_interest'] = 'correct_rejection'
        else:
            if not any(ind_button_presses > onset_current_stim):
                eeg_with_new_triggers.loc[onset_current_stim, 'triggers_of_interest'] = 'correct_rejection'

    # Mark hits and misses
    for itarget in range(len(ind_onsets_targets)):
        onset_current_target = ind_onsets_targets[itarget]
        if any(ind_stim_onsets > onset_current_target):
            onset_next_stim = ind_stim_onsets[np.min([i for i in range(len(ind_stim_onsets)) if ind_stim_onsets[i] > onset_current_target])]
            # Check if there is a button press between the current target and the next stimulus
            button_press_in_between = [onset_current_target < x < onset_next_stim for x in ind_button_presses]
            if any(button_press_in_between):
                ibutton_press = [i for i, b in enumerate(button_press_in_between) if b]
                eeg_with_new_triggers.loc[ind_button_presses[ibutton_press], 'triggers_of_interest'] = 'hit'
            else:
                eeg_with_new_triggers.loc[onset_current_target, 'triggers_of_interest'] = 'miss'
        else:
            if any(ind_button_presses > onset_current_target):
                ibutton_press = [i for i, b in enumerate(button_press_in_between) if b]
                eeg_with_new_triggers.loc[ind_button_presses[ibutton_press], 'triggers_of_interest'] = 'hit'
            else:
                eeg_with_new_triggers.loc[onset_current_target, 'triggers_of_interest'] = 'miss'

    # Mark false alarms
    for ibutton_press in range(len(ind_button_presses)):
        # Find preceding stimulus onset
        lst = list(ind_button_presses[ibutton_press] - ind_stim_onsets)
        non_negative_numbers = [num for num in lst if num >= 0]
        preceding_stimulus_onset = ind_stim_onsets[lst.index(min(non_negative_numbers))]
        if preceding_stimulus_onset in ind_onsets_nontargets:
            eeg_with_new_triggers.loc[ind_button_presses[ibutton_press], 'triggers_of_interest'] = 'false_alarm'

    return eeg_with_new_triggers

def detect_blinks(eeg, thresh_z_blinks, padding_blinks_samples, task, subject_id, visualize=False):

    if not (any(eeg['Fp1'].notnull()) and any(eeg['Fp1'].notnull())):
        print('NO DATA FOR FP1/FP2. NOT DETECTING BLINKS')
        return np.nan, np.nan
    else:
        # Note z-transformed frontal eeg channels
        eeg_frontal_z = eeg.copy()
        for channel in ['Fp1', 'Fp2']:
            eeg_frontal_z[channel] = (eeg_frontal_z[channel] - eeg_frontal_z[channel].mean()) / eeg_frontal_z[channel].std()
        if 'timestamp' not in eeg_frontal_z.columns:
            eeg_frontal_z.reset_index(inplace=True, drop=False)
        eeg_frontal_z = eeg_frontal_z[['timestamp', 'Fp1', 'Fp2']].reset_index()

        # Detect blinks based on Fp1 and Fp2 channels, note starts and ends
        idx_blinks = list(
            np.unique(sorted(list(eeg_frontal_z.loc[abs(eeg_frontal_z['Fp1']) > thresh_z_blinks].index) + list(
                eeg_frontal_z.loc[abs(eeg_frontal_z['Fp2']) > thresh_z_blinks].index))))
        if len(idx_blinks) == 0:
            blinks_starts_ends = []
            return [], []
        else:
            idx = 0
            idx_start_blink = idx_blinks[0]
            blinks_starts_ends = pd.DataFrame(columns=['blink_start', 'blink_end'])
            while idx < (len(idx_blinks) - 1):

                if idx_blinks[idx + 1] != idx_blinks[idx] + 1:
                    idx_end_blink = idx_blinks[idx]
                    blinks_starts_ends = blinks_starts_ends.append(
                        pd.DataFrame({'blink_start': [idx_start_blink], 'blink_end': [idx_end_blink]}))
                    idx_start_blink = idx_blinks[idx + 1]

                idx = idx + 1
            blinks_starts_ends = blinks_starts_ends.append(
                pd.DataFrame({'blink_start': [idx_start_blink], 'blink_end': [idx_blinks[idx]]}))
            blinks_starts_ends.reset_index(inplace=True, drop=True)

            # Add padding to blinks
            blinks_starts_ends_padded = blinks_starts_ends.copy()
            blinks_starts_ends_padded['blink_start'] = blinks_starts_ends_padded['blink_start'] - padding_blinks_samples
            blinks_starts_ends_padded['blink_end'] = blinks_starts_ends_padded['blink_end'] + padding_blinks_samples
            for iblink in range(blinks_starts_ends_padded.shape[0]):
                if blinks_starts_ends_padded.loc[iblink, 'blink_start'] < 0:
                    blinks_starts_ends_padded.loc[iblink, 'blink_start'] = 0
                if blinks_starts_ends_padded.loc[iblink, 'blink_end'] > (len(eeg_frontal_z) - 1):
                    blinks_starts_ends_padded.loc[iblink, 'blink_end'] = len(eeg_frontal_z) - 1
            blinks_starts_ends_padded['blink_start'] = blinks_starts_ends_padded['blink_start'].astype(int)
            blinks_starts_ends_padded['blink_end'] = blinks_starts_ends_padded['blink_end'].astype(int)

            blinks_starts_ends_samples = blinks_starts_ends_padded.copy()

            # Merge overlapping blinks
            if blinks_starts_ends_samples.shape[0] > 1:
                blinks_starts_ends_samples_merged = []
                # Initialize variables to track the current merged period
                current_start = blinks_starts_ends_samples.iloc[0]['blink_start']
                current_end = blinks_starts_ends_samples.iloc[0]['blink_end']
                for index, row in blinks_starts_ends_samples.iterrows():
                    start = row['blink_start']
                    end = row['blink_end']

                    # Check if the current period overlaps or is contained in the current merged period
                    if start <= current_end:
                        # Update the current merged period
                        current_end = max(current_end, end)
                    else:
                        # Append the current merged period to the list and start a new merged period
                        blinks_starts_ends_samples_merged.append({'blink_start': current_start, 'blink_end': current_end})
                        current_start = start
                        current_end = end
                blinks_starts_ends_samples = pd.DataFrame(blinks_starts_ends_samples_merged)

            # Note blink timestamps
            blinks_timestamps = blinks_starts_ends_samples.copy()
            for iblink in range(blinks_starts_ends_samples.shape[0]):
                blinks_timestamps.loc[iblink, 'blink_start'] = eeg_frontal_z.loc[
                    blinks_starts_ends_samples.loc[iblink, 'blink_start'], 'timestamp']
                blinks_timestamps.loc[iblink, 'blink_end'] = eeg_frontal_z.loc[
                    blinks_starts_ends_samples.loc[iblink, 'blink_end'], 'timestamp']

            # Optional: Visualize for control
            if visualize:
                fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
                axs = axs.flatten()
                axs[0].plot(eeg_frontal_z['timestamp'], eeg_frontal_z['Fp1'])
                for iblink in range(blinks_timestamps.shape[0]):
                    axs[0].axvspan(blinks_timestamps.loc[iblink, 'blink_start'],
                                   blinks_timestamps.loc[iblink, 'blink_end'], alpha=0.5, color='red')
                axs[0].set_title('Fp1')
                axs[1].plot(eeg_frontal_z['timestamp'], eeg_frontal_z['Fp2'])
                for iblink in range(blinks_timestamps.shape[0]):
                    axs[1].axvspan(blinks_timestamps.loc[iblink, 'blink_start'],
                                   blinks_timestamps.loc[iblink, 'blink_end'], alpha=0.5, color='red')
                axs[1].set_title('Fp2')
                fig.suptitle(f'{task}, S{subject_id}, z-threshold={thresh_z_blinks}, Fp2')

            return blinks_starts_ends_samples, blinks_timestamps

def compute_blinkrate_sliding_window(blinks_timestamps, sliding_window_duration, sliding_window_overlap, block_duration):
    blinkrate_moving_windows = pd.DataFrame(columns=['center_window', 'blink_rate'])
    # Compute blink rates in moving windows
    start_window = 0
    end_window = sliding_window_duration
    while end_window < block_duration:
        blinks_within_window = blinks_timestamps[
            ((blinks_timestamps['blink_start'] >= start_window) & (blinks_timestamps['blink_start'] <= end_window)) |
            ((blinks_timestamps['blink_end'] >= start_window) & (blinks_timestamps['blink_end'] <= end_window)) |
            ((blinks_timestamps['blink_start'] <= start_window) & (blinks_timestamps['blink_end'] >= end_window))]
        nblinks_in_window = blinks_within_window.shape[0]
        center_window = int(start_window + (sliding_window_duration / 2))  # compute center of the window in seconds

        blinkrate_moving_windows = pd.concat([blinkrate_moving_windows, pd.DataFrame({'center_window': [center_window],
                                                                                      'blink_rate': [
                                                                                          nblinks_in_window]})])

        start_window += int(sliding_window_duration * sliding_window_overlap)
        end_window = start_window + sliding_window_duration

    return blinkrate_moving_windows