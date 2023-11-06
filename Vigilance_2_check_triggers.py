#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:28:14 2022

@author: nicolaiwolpert
"""

import numpy as np
import pandas as pd

import math
import os
from os import listdir
from os.path import isfile, join
import glob
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

root = os.path.dirname(os.getcwd())
root_dir = root + '/Vigilance/'
raw_dir = root_dir + '/Data_raw/'

# Specify task ('oddball', 'atc', 'line_task_sim', or 'line_task_succ')
task = 'line_task_sim'

# Specify subject ID
subject_id = '30'

filepath_eeg = raw_dir + f'/{task}/EEG/{subject_id}/vigilance_{task}_EEG_{subject_id}.csv'
EEG_data = pd.read_csv(filepath_eeg)

def read_behavior_files(subject_id, block_type='all'):
    """read_behavior_files: Reads in the behavioral results file for a given subject and merges them into one"""

    # Get a list of CSV file paths in the directory
    csv_files = glob.glob(raw_dir + task + '\\Behavior\\' + subject_id + '\\' + '*.csv')
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

    if block_type == 'experiment':
        results_behavior = results_behavior.loc[results_behavior.block_type == 'experiment']
        results_behavior.reset_index(inplace=True, drop=True)

    return results_behavior

### Specify the experimental triggers (copy of 'triggers_vigilance' in eegnb)
TRIGGERS = {
    ##-Task events
    # General
    'TRIGGER_START_EXPERIMENT': 1,
    'TRIGGER_END_EXPERIMENT': 2,
    'TRIGGER_START_TRAINING': 3,
    'TRIGGER_END_TRAINING': 4,
    'TRIGGER_START_PRACTICE': 5,
    'TRIGGER_END_PRACTICE': 6,
    'TRIGGER_START_BLOCK': 10,
    'TRIGGER_END_BLOCK': 11,
    'TRIGGER_START_TRIAL': 20,
    'TRIGGER_END_TRIAL': 21,
    'TRIGGER_ONSET_FEEDBACK': 60,
    'TRIGGER_OFFSET_FEEDBACK': 61,
    'TRIGGER_BUTTON_PRESS': 7,

    # Oddball-specific
    'TRIGGER_ONSET_TARGET': 30,
    'TRIGGER_OFFSET_TARGET': 31,
    'TRIGGER_ONSET_DISTRACTOR_NONTARGET': 40,
    'TRIGGER_OFFSET_DISTRACTOR_NONTARGET': 41,
    'TRIGGER_ONSET_DISTRACTOR_STANDARD': 50,
    'TRIGGER_OFFSET_DISTRACTOR_STANDARD': 51,

    # ATC & line task
    'TRIGGER_ONSET_CRITICAL': 30,
    'TRIGGER_OFFSET_CRITICAL': 31,
    'TRIGGER_ONSET_NONCRITICAL' : 40,
    'TRIGGER_OFFSET_NONCRITICAL' : 41
}

# Initialize variables to store the indices for each trigger type
ind_trigger_start_experiment = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_START_EXPERIMENT']].index)
ind_trigger_end_experiment = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_END_EXPERIMENT']].index)
ind_trigger_start_training = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_START_TRAINING']].index)
ind_trigger_end_training = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_END_TRAINING']].index)
ind_trigger_start_practice = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_START_PRACTICE']].index)
ind_trigger_end_practice = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_END_PRACTICE']].index)
ind_trigger_start_block = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_START_BLOCK']].index)
ind_trigger_end_block = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_END_BLOCK']].index)
ind_trigger_start_trial = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_START_TRIAL']].index)
ind_trigger_end_trial = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_END_TRIAL']].index)
ind_trigger_onset_target = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_ONSET_TARGET']].index)
ind_trigger_offset_target = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_OFFSET_TARGET']].index)

if task == 'oddball':
    ind_trigger_onset_distractor_nontarget = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_ONSET_DISTRACTOR_NONTARGET']].index)
    ind_trigger_offset_distractor_nontarget = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_OFFSET_DISTRACTOR_NONTARGET']].index)
    ind_trigger_onset_distractor_standard = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_ONSET_DISTRACTOR_STANDARD']].index)
    ind_trigger_offset_distractor_standard = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_OFFSET_DISTRACTOR_STANDARD']].index)
    ind_trigger_stimulus_onsets = sorted(ind_trigger_onset_target + ind_trigger_onset_distractor_nontarget +
                                         ind_trigger_onset_distractor_standard)
    ind_trigger_stimulus_offsets = sorted(ind_trigger_offset_target + ind_trigger_offset_distractor_nontarget +
                                         ind_trigger_offset_distractor_standard)
elif (task == 'atc') or (task.startswith('line_task')):
    ind_trigger_onset_critical = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_ONSET_CRITICAL']].index)
    ind_trigger_offset_critical = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_OFFSET_CRITICAL']].index)
    ind_trigger_onset_noncritical = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_ONSET_NONCRITICAL']].index)
    ind_trigger_offset_noncritical = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_OFFSET_NONCRITICAL']].index)
    ind_trigger_stimulus_onsets = sorted(ind_trigger_onset_critical + ind_trigger_onset_noncritical)
    ind_trigger_stimulus_offsets = sorted(ind_trigger_offset_critical + ind_trigger_offset_noncritical)
ind_trigger_onset_feedback = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_ONSET_FEEDBACK']].index)
ind_trigger_offset_feedback = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_OFFSET_FEEDBACK']].index)
ind_trigger_button_press = list(EEG_data.loc[EEG_data['trigger'] == TRIGGERS['TRIGGER_BUTTON_PRESS']].index)

# Print the number of triggers for each type
print(f"Number of TRIGGER_START_EXPERIMENT: {len(ind_trigger_start_experiment)}")
print(f"Number of TRIGGER_END_EXPERIMENT: {len(ind_trigger_end_experiment)}")
print(f"Number of TRIGGER_START_TRAINING: {len(ind_trigger_start_training)}")
print(f"Number of TRIGGER_END_TRAINING: {len(ind_trigger_end_training)}")
print(f"Number of TRIGGER_START_PRACTICE: {len(ind_trigger_start_practice)}")
print(f"Number of TRIGGER_END_PRACTICE: {len(ind_trigger_end_practice)}")
print(f"Number of TRIGGER_START_BLOCK: {len(ind_trigger_start_block)}")
print(f"Number of TRIGGER_END_BLOCK: {len(ind_trigger_end_block)}")
print(f"Number of TRIGGER_START_TRIAL: {len(ind_trigger_start_trial)}")
print(f"Number of TRIGGER_END_TRIAL: {len(ind_trigger_end_trial)}")
print(f"Number of stimulus onsets in general: {len(ind_trigger_stimulus_onsets)}")
print(f"Number of stimulus offsets in general: {len(ind_trigger_stimulus_offsets)}")
if task == 'oddball':
    print(f"Number of TRIGGER_ONSET_TARGET: {len(ind_trigger_onset_target)}")
    print(f"Number of TRIGGER_OFFSET_TARGET: {len(ind_trigger_offset_target)}")
    print(f"Number of TRIGGER_ONSET_DISTRACTOR_NONTARGET: {len(ind_trigger_onset_distractor_nontarget)}")
    print(f"Number of TRIGGER_OFFSET_DISTRACTOR_NONTARGET: {len(ind_trigger_offset_distractor_nontarget)}")
    print(f"Number of TRIGGER_ONSET_DISTRACTOR_STANDARD: {len(ind_trigger_onset_distractor_standard)}")
    print(f"Number of TRIGGER_OFFSET_DISTRACTOR_STANDARD: {len(ind_trigger_offset_distractor_standard)}")
elif (task == 'atc') or task == 'line_task':
    print(f"Number of TRIGGER_ONSET_CRITICAL: {len(ind_trigger_onset_critical)}")
    print(f"Number of TRIGGER_OFFSET_CRITICAL: {len(ind_trigger_offset_critical)}")
    print(f"Number of TRIGGER_ONSET_NONCRITICAL: {len(ind_trigger_onset_noncritical)}")
    print(f"Number of TRIGGER_OFFSET_NONCRITICAL: {len(ind_trigger_offset_noncritical)}")
print(f"Number of TRIGGER_ONSET_FEEDBACK: {len(ind_trigger_onset_feedback)}")
print(f"Number of TRIGGER_OFFSET_FEEDBACK: {len(ind_trigger_offset_feedback)}")
print(f"Number of TRIGGER_BUTTON_PRESS: {len(ind_trigger_button_press)}")

### Read in behavioral results
results_behavior = read_behavior_files(subject_id, 'all')

### Stimulus timings
stimulus_durations = [EEG_data.loc[ind_trigger_stimulus_offsets[i], 'timestamp'] - EEG_data.loc[ind_trigger_stimulus_onsets[i], 'timestamp']
                      for i in range(len(ind_trigger_stimulus_onsets))]
print('##### Actual stimulus durations from triggers:')
print(stimulus_durations)

### Reaction time

RTs_triggers = []
for ibutton_press in range(len(ind_trigger_button_press)):
    timestamp_button_press = EEG_data.loc[ind_trigger_button_press[ibutton_press], 'timestamp']
    # find timestamp of preceding stimulus onset
    timestamps_trigger_stimulus_onsets = [EEG_data.loc[ind_trigger_stimulus_onsets[istim], 'timestamp'] for istim in range(len(ind_trigger_stimulus_onsets))]

    timestamp_stim_onset_preceding = None
    for timestamp_stim_onset in timestamps_trigger_stimulus_onsets:
        if timestamp_stim_onset >= timestamp_button_press:
            break
        timestamp_stim_onset_preceding = timestamp_stim_onset

    RTs_triggers = RTs_triggers + [timestamp_button_press - timestamp_stim_onset_preceding]

nbutton_presses_behavior = results_behavior['rt'].dropna().shape[0]
nbutton_presses_triggers = len(RTs_triggers)
if results_behavior['rt'].dropna().shape[0] != len(RTs_triggers):
    warnings.warn('Number of button presses does not match between behavior file ({}) and triggers ({})!'
                     .format(nbutton_presses_behavior, nbutton_presses_triggers))
rt_comparison = pd.DataFrame(columns=['behavior_file','triggers'])
rt_comparison['behavior_file'] = results_behavior['rt'].dropna()
rt_comparison['triggers'] = RTs_triggers
rt_comparison['difference'] = rt_comparison['behavior_file'] - rt_comparison['triggers']
pd.set_option('display.max_rows', 30)
print(rt_comparison)

plt.figure()
plt.scatter(rt_comparison['behavior_file'], rt_comparison['triggers'])
plt.plot([0, rt_comparison.max().max()*1.1], [0, rt_comparison.max().max()*1.1], color='red', linestyle='--')
plt.xlabel('RT according to behavior file')
plt.ylabel('RT according to triggers')
plt.title(task + ', S' + subject_id)
plt.show()

