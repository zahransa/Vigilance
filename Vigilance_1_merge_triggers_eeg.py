#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:28:14 2022

@author: nicolaiwolpert
"""

import numpy as np
import pandas as pd
import math
import re
import warnings
import os
from time import time
import warnings

start_time = time()

print('######### Merging triggers with EEG file (this will take a while)... #########')
root = os.path.dirname(os.getcwd())

# Specify task ('oddball', 'atc', 'line_task_sim', or 'line_task_succ')
task = 'line_task_sim'

# Specify subject ID
subject_id = '30'

# Specify where original data are located
dir_eeg_orig = root + '\\Vigilance\\Data_raw\\' + task + '\\EEG_without_triggers\\'
dir_triggers = root + '\\Vigilance\\Data_raw\\' + task + '\\Triggers\\'

# specify where data should be saved
data_dir_save = root + '\\Vigilance\\Data_raw\\' + task + '\\EEG\\' + subject_id + '\\'

triggers_filename = 'triggers_subject30_2023-09-15-15.08.csv'
triggers_saved_file = dir_triggers + triggers_filename

### Look for corresponding EEG file

pattern = r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.csv'
match = re.search(pattern, triggers_saved_file)
file_ending = match.group()
eeg_found = False
for filename in os.listdir(dir_eeg_orig):
    if filename.startswith('EEGdata') and filename.endswith(file_ending) and subject_id in filename:
        EEGdata_saved_file = os.path.join(dir_eeg_orig, filename)
        print('EEG file found:', EEGdata_saved_file)
        eeg_found = True
        break
if not eeg_found:
    raise ValueError('Corresponding EEG file not found for {}. Cannot insert triggers.'.format(triggers_saved_file))

triggers_data = pd.read_csv(triggers_saved_file)
EEG_data = pd.read_csv(EEGdata_saved_file, on_bad_lines='skip')


def truncate(number, digits) -> float:
    # function that easily truncates the used value after the wanted number of decimal digits
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


# create the empty lists to fill in with values from loops comparisons
stims = []
index_triggers = []

# - Comparison loops to find the right index in EEG data that corresponds to the psychopy triggers

timestamps_trigger_vs_eeg = pd.DataFrame(columns=('timestamp_trigger', 'timestamp_matched_EEG',
                                                  'closest_timestamp_eeg', 'found_by_truncate'))
timestamps_trigger_vs_eeg['itrigger'] = list(range(len(triggers_data)))
timestamps_trigger_vs_eeg['trigger_type'] = triggers_data.stim
for i in range(len(triggers_data)):
    print('Searching for closest timestamp for trigger {}, type {}...'.format(i, triggers_data['stim'][i]))

    # Extract EEG data around the current trigger timestamp to reduce running time
    EEG_data_around = EEG_data.loc[(EEG_data.timestamps >= (triggers_data.timestamps[i] - 20)) &
                                   (EEG_data.timestamps <= (triggers_data.timestamps[i] + 20))]
    if EEG_data_around.shape[0] == 0:
        print('WARNING: Found no EEG around trigger {} timestamp'.format(i))
        closest_timestamp = np.nan
    else:
        ind_closest = EEG_data_around.iloc[(EEG_data_around['timestamps'] - triggers_data.timestamps[i]).abs().argsort()[:1]].index
        closest_timestamp = EEG_data_around.loc[ind_closest, 'timestamps'].values[0]
        timestamps_trigger_vs_eeg.loc[i, 'closest_timestamp_eeg'] = closest_timestamp

    found_matching_timestamp_EEG = False
    for j in EEG_data_around.index:
        if truncate(EEG_data_around.timestamps[j], 3) == truncate(triggers_data.timestamps[i], 3):
            stims.append(triggers_data.stim[i])
            index_triggers.append(j)
            found_matching_timestamp_EEG = True
            timestamps_trigger_vs_eeg.loc[i, 'timestamp_matched_EEG'] = EEG_data_around.timestamps[j]
            timestamps_trigger_vs_eeg.loc[i, 'timestamp_trigger'] = triggers_data.timestamps[i]
            timestamps_trigger_vs_eeg.loc[i, 'found_by_truncate'] = True
            print('Found.')
            break
        elif truncate(EEG_data_around.timestamps[j], 2) == truncate(triggers_data.timestamps[i], 2):
            stims.append(triggers_data.stim[i])
            index_triggers.append(j + 1)
            found_matching_timestamp_EEG = True
            timestamps_trigger_vs_eeg.loc[i, 'timestamp_matched_EEG'] = EEG_data_around.timestamps[j]
            timestamps_trigger_vs_eeg.loc[i, 'timestamp_trigger'] = triggers_data.timestamps[i]
            timestamps_trigger_vs_eeg.loc[i, 'found_by_truncate'] = True
            print('Found.')
            break
    if not found_matching_timestamp_EEG:
        timestamps_trigger_vs_eeg.loc[i, 'found_by_truncate'] = False
        print('WARNING: Could not find matching timestamp for trigger {}, timestamp {}!!!'
                         '\nClosest timestamp found in EEG: {}'.format(triggers_data.stim[i], triggers_data.timestamps[i], closest_timestamp))

timestamps_trigger_vs_eeg['difference_timestamps'] = timestamps_trigger_vs_eeg['timestamp_trigger'] - timestamps_trigger_vs_eeg['timestamp_matched_EEG']

import matplotlib.pyplot as plt
plt.figure()
plt.hist(timestamps_trigger_vs_eeg['difference_timestamps'])
plt.title('Difference in trigger timestamps trigger file vs. EEG')
plt.xlabel('seconds')
plt.show()

print('Done with searching for matching timestamps in EEG')

# - loops allowing to create a list/column with the trigger IDs (events) at the right positions/index, with the same shape/length as the EEG data

triggers_index = []

for i in range(len(index_triggers)):

    if i == 0:
        for n in range(0, (index_triggers[i])):
            triggers_index.append('')

        triggers_index.append(stims[i])

    else:
        for n in range(index_triggers[i - 1] + 1, index_triggers[i]):
            triggers_index.append('')

        triggers_index.append(stims[i])

last_trigger = len(triggers_index)

for i in range(last_trigger, len(EEG_data)):
    triggers_index.append('')

# - Create the full data set including EEG data and triggers and save it

stim = pd.DataFrame(triggers_index)
stim_new = stim.rename(columns={0: 'trigger'})
full_data = pd.concat([EEG_data, stim_new], axis=1)
full_data = full_data.rename(columns={'timestamps': 'timestamp'})
full_data['timestamp'] -= full_data['timestamp'][0]  # Set first timestamp to zero

# For atc, subject 24, some negative timestamps were found (reason unknown). Correct this
if (task=='atc') and (subject_id=='24'):
    print('CORRECTING NEGATIVE TIMESTAMPS...')

    ind_negative_timestamps = full_data.loc[full_data.timestamp<0].index
    n_negative_timestamps = len(ind_negative_timestamps)
    last_correct_timestamp_before = full_data.loc[ind_negative_timestamps[0] - 1, 'timestamp']
    first_correct_timestamp_after = full_data.loc[ind_negative_timestamps[-1] + 1, 'timestamp']

    timestamps_corrected = np.linspace(last_correct_timestamp_before+0.004, first_correct_timestamp_after-0.004, n_negative_timestamps)
    full_data.loc[ind_negative_timestamps, 'timestamp'] = timestamps_corrected

if os.path.isfile(data_dir_save + 'vigilance_' + task + '_EEG_' + subject_id + '.csv'):
    warnings.warn('EEG file exists already for ID {}! Not saving'.format(subject_id))
else:
    if not os.path.isdir(data_dir_save):
        os.makedirs(data_dir_save)
    full_data.to_csv(data_dir_save + 'vigilance_' + task + '_EEG_' + subject_id + '.csv', index=False)

print("--- Triggers merged. Process took %s minutes ---" % (round((time() - start_time) / 60, 2)))