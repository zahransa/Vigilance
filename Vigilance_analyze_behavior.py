import numpy as np
import pandas as pd
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import norm
from collections import Counter
import glob
import warnings
warnings.filterwarnings("ignore")

from Scripts.vigilance_tools import *

#%matplotlib

plt.close('all')

root = os.path.dirname(os.getcwd())
root = os.path.join(root, 'Vigilance')

raw_dir = os.path.join(root, 'Data_raw')

### Overall distribution of performance

# Specify subjects for the different tasks with clean EEG & behavior
tasks_subjects = {'atc': ['01', '03', '04', '05', '08', '15', '21', '23', '24'],
                  'line_task_sim': ['02', '03', '04', '18', '25'],
                  'line_task_succ': ['02', '04', '05', '07'],
                  'oddball': ['04', '07']}

distribution_performance = pd.DataFrame(columns=['task_subject','nhits','hit_rate','accuracy','nfalse_alarms','mean_rt_target'])
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        results_behavior = read_behavior_files(subject_id, task, raw_dir, block_type='experiment')

        performance_indeces = compute_performance_indeces(results_behavior)
        distribution_performance = pd.concat([distribution_performance,
                                              pd.DataFrame({'task_subject': [task + '_' + subject_id],
                                                            'nhits': [performance_indeces['nhits'][0]],
                                                            'nmisses': [performance_indeces['nmisses'][0]],
                                                            'hit_rate': [performance_indeces['hit_rate'][0]],
                                                            'accuracy': [performance_indeces['accuracy'][0]],
                                                            'nfalse_alarms': [performance_indeces['nfalse_alarms'][0]],
                                                            'mean_rt_target': [performance_indeces['mean_rt_target'][0]]})])

distribution_performance['ratio_false_alarms_button_presses'] = distribution_performance['nfalse_alarms'] / (distribution_performance['nhits'] + distribution_performance['nfalse_alarms'])
distribution_performance.reset_index(inplace=True, drop=True)

# Compute BIS, based on aggregates over participants
accuracies_z = [(accuracy - np.mean(distribution_performance['accuracy'])) / np.std(distribution_performance['accuracy']) for accuracy in distribution_performance['accuracy']]
RT_z = [(rt - np.mean(distribution_performance['mean_rt_target'])) / np.std(distribution_performance['mean_rt_target']) for rt in distribution_performance['mean_rt_target']]
BIS = [accuracies_z[i] - RT_z[i] for i in range(len(accuracies_z))]
distribution_performance['BIS'] = BIS

colors = matplotlib.cm.tab20(range(distribution_performance.shape[0]))

fig, axs = plt.subplots(1, 6)
axs = axs.flatten()
for i in range(distribution_performance.shape[0]):
    axs[0].scatter(1, distribution_performance.loc[i, 'hit_rate'], edgecolors=colors[i], facecolors='none', s=40, label=distribution_performance.loc[i, 'task_subject'])
axs[0].set_xticks([1])
axs[0].set_xticklabels([''])
axs[0].set_ylabel('hit rate')
for i in range(distribution_performance.shape[0]):
    axs[1].scatter(1, distribution_performance.loc[i, 'accuracy'], edgecolors=colors[i], facecolors='none', s=40, label=distribution_performance.loc[i, 'task_subject'])
axs[1].set_xticks([1])
axs[1].set_xticklabels([''])
axs[1].set_ylabel('accuracy')
for i in range(distribution_performance.shape[0]):
    axs[2].scatter(1, distribution_performance.loc[i, 'nfalse_alarms'], edgecolors=colors[i], facecolors='none', s=40, label=distribution_performance.loc[i, 'task_subject'])
axs[2].set_xticks([1])
axs[2].set_xticklabels([''])
axs[2].set_ylabel('nfalse_alarms')
for i in range(distribution_performance.shape[0]):
    axs[3].scatter(1, distribution_performance.loc[i, 'ratio_false_alarms_button_presses'], edgecolors=colors[i], facecolors='none', s=40, label=distribution_performance.loc[i, 'task_subject'])
axs[3].set_xticks([1])
axs[3].set_xticklabels([''])
axs[3].set_ylabel('ratio false alarms to button presses')
for i in range(distribution_performance.shape[0]):
    axs[4].scatter(1, distribution_performance.loc[i, 'mean_rt_target'], edgecolors=colors[i], facecolors='none', s=40, label=distribution_performance.loc[i, 'task_subject'])
axs[4].set_xticks([1])
axs[4].set_xticklabels([''])
axs[4].set_ylabel('mean_rt_target')
for i in range(distribution_performance.shape[0]):
    axs[5].scatter(1, distribution_performance.loc[i, 'BIS'], edgecolors=colors[i], facecolors='none', s=40, label=distribution_performance.loc[i, 'task_subject'])
axs[5].set_xticks([1])
axs[5].set_xticklabels([''])
axs[5].set_ylabel('BIS')
plt.legend()
plt.tight_layout()

distribution_performance[['task_subject', 'nfalse_alarms', 'ratio_false_alarms_button_presses']]
distribution_performance[['task_subject', 'hit_rate', 'accuracy']]

### One subject of interest

# Specify task ('oddball', 'atc', 'line_task_sim' or 'line_task_succ')
task = 'line_task_sim'

# Specify subject of interest
subject_id = '25'

results_behavior = read_behavior_files(subject_id, task, raw_dir, block_type = 'experiment')

performance_indeces_bin1, performance_indeces_bin2, performance_indeces_bin3, performance_indeces_bin4 = compute_performance_per_bin(results_behavior, task)

# Choose performance index
performance_indeces_of_interest = ['hit_rate', 'dprime', 'criterion', 'mean_rt_target', 'accuracy', 'nfalse_alarms', 'ies']
for perf_ind in performance_indeces_of_interest:
    plt.figure()
    perf_by_block = [performance_indeces_bin1[perf_ind][0], performance_indeces_bin2[perf_ind][0],
                     performance_indeces_bin3[perf_ind][0], performance_indeces_bin4[perf_ind][0]]
    plt.scatter(list(range(1, 5)), perf_by_block, color='blue')
    plt.plot(list(range(1, 5)), perf_by_block, color='blue')
    plt.xticks(list(range(1, 5)))
    plt.xlabel('Bin')
    if perf_ind == 'ies':
        plt.ylabel('inverse efficiency score')
    else:
        plt.ylabel(perf_ind)
    plt.title(task + ', subject ' + subject_id)
    plt.show()

### Multiple subjects in one plot

# plt.close('all')

subjects_atc = ['02', '04', '05', '07', '08', '13', '15']
subjects_line_task_sim = ['02', '03', '05', '07', '12', '18', '25']
subjects_line_task_succ = ['02', '04', '05', '06', '07']
subjects_oddball = ['01', '04']

# Normalize or not
normalize = False

# Select performance index of interest
perf_indeces = ['hit_rate', 'accuracy', 'nfalse_alarms', 'nmisses', 'mean_rt_target', 'ies']

for task in ['atc', 'line_task_sim', 'line_task_succ', 'oddball']:

    if task == 'atc':
        subjects = subjects_atc
    elif task == 'line_task_sim':
        subjects = subjects_line_task_sim
    elif task == 'line_task_succ':
        subjects = subjects_line_task_succ
    elif task == 'oddball':
        subjects = subjects_oddball

    # Get maximally distinguishable colors
    colors = matplotlib.cm.tab20(range(len(subjects)))

    fig, axs = plt.subplots(int(len(perf_indeces)/2), 2)
    axs = axs.flatten()
    for iperf in range(len(perf_indeces)):
        perf_index = perf_indeces[iperf]
        for isubject in range(len(subjects)):

            results_behavior = read_behavior_files(subjects[isubject], task, raw_dir, block_type = 'experiment')
            performance_indeces_bin1, performance_indeces_bin2, performance_indeces_bin3, performance_indeces_bin4 = compute_performance_per_bin(results_behavior, task)

            perf_by_bin = [performance_indeces_bin1[perf_index][0], performance_indeces_bin2[perf_index][0],
                           performance_indeces_bin3[perf_index][0], performance_indeces_bin4[perf_index][0]]

            if normalize:
                perf_by_bin = [(p - np.nanmin(perf_by_bin)) / (np.nanmax(perf_by_bin) - np.nanmin(perf_by_bin)) for p in perf_by_bin]
            for ibin in range(len(perf_by_bin)):
                if not math.isnan(perf_by_bin[ibin]):
                    axs[iperf].scatter(ibin+1, perf_by_bin[ibin], color=colors[isubject])
                    axs[iperf].plot(ibin+1, perf_by_bin[ibin], color=colors[isubject])
            axs[iperf].plot(list(range(1, 5)), perf_by_bin, color=colors[isubject], label='S' + subjects[isubject])

        axs[iperf].set_xticks(list(range(1, 5)))
        axs[iperf].set_xlabel('Bin')
        if perf_index == 'ies':
            if normalize:
                axs[iperf].set_ylabel('inverse efficiency score, norm.')
            else:
                axs[iperf].set_ylabel('inverse efficiency score')
        else:
            if normalize:
                axs[iperf].set_ylabel(perf_index + ', norm.')
            else:
                axs[iperf].set_ylabel(perf_index)
        if iperf == (len(perf_indeces)-1):
            axs[iperf].legend()
    plt.suptitle(task)
    plt.tight_layout()
    plt.show()

######################################### Compute and show performance per bin #########################################

performances_bin1 = pd.DataFrame()
performances_bin2 = pd.DataFrame()
performances_bin3 = pd.DataFrame()
performances_bin4 = pd.DataFrame()
for task in tasks_subjects.keys():
    for subject_id in tasks_subjects[task]:

        results_behavior = read_behavior_files(subject_id, task, raw_dir, block_type='experiment')

        performance_indeces_bin1, performance_indeces_bin2, performance_indeces_bin3, performance_indeces_bin4 = compute_performance_per_bin(results_behavior, task)

        performance_indeces_bin1.insert(loc=0, column='task_subject', value= task + '_' + subject_id)
        performances_bin1 = pd.concat([performances_bin1, performance_indeces_bin1])

        performance_indeces_bin2.insert(loc=0, column='task_subject', value= task + '_' + subject_id)
        performances_bin2 = pd.concat([performances_bin2, performance_indeces_bin2])

        performance_indeces_bin3.insert(loc=0, column='task_subject', value= task + '_' + subject_id)
        performances_bin3 = pd.concat([performances_bin3, performance_indeces_bin3])

        performance_indeces_bin4.insert(loc=0, column='task_subject', value= task + '_' + subject_id)
        performances_bin4 = pd.concat([performances_bin4, performance_indeces_bin4])

accuracies_z = [(accuracy - np.mean(performances_bin1['accuracy'])) / np.std(performances_bin1['accuracy']) for accuracy in performances_bin1['accuracy']]
RT_z = [(rt - np.mean(performances_bin1['mean_rt_target'])) / np.std(performances_bin1['mean_rt_target']) for rt in performances_bin1['mean_rt_target']]
BIS_bin1 = [accuracies_z[i] - RT_z[i] for i in range(len(accuracies_z))]

accuracies_z = [(accuracy - np.mean(performances_bin2['accuracy'])) / np.std(performances_bin2['accuracy']) for accuracy in performances_bin2['accuracy']]
RT_z = [(rt - np.mean(performances_bin2['mean_rt_target'])) / np.std(performances_bin2['mean_rt_target']) for rt in performances_bin2['mean_rt_target']]
BIS_bin2 = [accuracies_z[i] - RT_z[i] for i in range(len(accuracies_z))]

accuracies_z = [(accuracy - np.mean(performances_bin3['accuracy'])) / np.std(performances_bin3['accuracy']) for accuracy in performances_bin3['accuracy']]
RT_z = [(rt - np.mean(performances_bin3['mean_rt_target'])) / np.std(performances_bin3['mean_rt_target']) for rt in performances_bin3['mean_rt_target']]
BIS_bin3 = [accuracies_z[i] - RT_z[i] for i in range(len(accuracies_z))]

accuracies_z = [(accuracy - np.mean(performances_bin4['accuracy'])) / np.std(performances_bin4['accuracy']) for accuracy in performances_bin4['accuracy']]
RT_z = [(rt - np.mean(performances_bin4['mean_rt_target'])) / np.std(performances_bin4['mean_rt_target']) for rt in performances_bin4['mean_rt_target']]
BIS_bin4 = [accuracies_z[i] - RT_z[i] for i in range(len(accuracies_z))]

performances_bin1['bis'] = BIS_bin1
performances_bin2['bis'] = BIS_bin2
performances_bin3['bis'] = BIS_bin3
performances_bin4['bis'] = BIS_bin4

# Choose performance index
performance_indeces_of_interest = ['mean_rt_target', 'accuracy', 'nfalse_alarms', 'bis', 'ies']
colors = matplotlib.cm.tab20(range(distribution_performance.shape[0]))
for perf_ind in performance_indeces_of_interest:
    plt.figure()
    icolor = 0
    for task in tasks_subjects.keys():
        for subject_id in tasks_subjects[task]:
            perf_by_block = [performances_bin1.loc[performances_bin1.task_subject==task + '_' + subject_id, perf_ind][0],
                             performances_bin2.loc[performances_bin2.task_subject == task + '_' + subject_id, perf_ind][0],
                             performances_bin3.loc[performances_bin3.task_subject == task + '_' + subject_id, perf_ind][0],
                             performances_bin4.loc[performances_bin4.task_subject == task + '_' + subject_id, perf_ind][0]]
            plt.scatter(list(range(1, 5)), perf_by_block, color=colors[icolor])
            plt.plot(list(range(1, 5)), perf_by_block, color=colors[icolor])
            plt.xticks(list(range(1, 5)))
            plt.xlabel('Bin')
            if perf_ind == 'ies':
                plt.ylabel('inverse efficiency score')
            elif perf_ind == 'bis':
                plt.ylabel('ibalanced integration score')
            else:
                plt.ylabel(perf_ind)
            plt.show()
            icolor += 1


