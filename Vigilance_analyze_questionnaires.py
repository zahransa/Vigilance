import numpy as np
import pandas as pd
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings("ignore")

#%matplotlib

plt.close('all')

root = os.path.dirname(os.getcwd())
root = os.path.join(root, 'Vigilance')

raw_dir = os.path.join(root, 'Data_raw')
eeg_dir = os.path.join(raw_dir, 'EEG')
behavior_dir = os.path.join(raw_dir, 'Behavior')
data_work_path = root + '/Data_work/'

questionnaires = pd.read_csv(raw_dir + '/questionnaires_answers.csv', sep=';', dtype={'Subject ID': str})
questionnaires = questionnaires.dropna().reset_index(drop=True)
questionnaires["subject_id"] = questionnaires["subject_id"].astype(str).str.zfill(2)
questionnaires.dropna(inplace=True)

### Differences in questionnaires between experiments

# List of variables of interest for the analysis
variables_of_interest = ['NASA_mental', 'NASA_physical', 'NASA_temporal',
                         'NASA_performance', 'NASA_effort', 'NASA_frustration',
                         'NASA_stress', 'NASA_vigilance_overall',
                         'NASA_vigilance1', 'NASA_vigilance2']

# Perform repeated measures ANOVA for each variable
for variable in variables_of_interest:
    formula = f'{variable} ~ C(task)'
    model = ols(formula, data=questionnaires).fit()
    print(f"\nVariable: {variable}")
    print(model.summary())







### Plot difference in vigilance ratings between first and second half
# Get maximally distinguishable colors
colors = matplotlib.cm.tab20(range(questionnaires.shape[0]))
fig, ax = plt.subplots()
# plt.scatter([1]*questionnaires.shape[0], questionnaires['NASA_vigilance1'])
# plt.scatter([2]*questionnaires.shape[0], questionnaires['NASA_vigilance2'])
for i in range(questionnaires.shape[0]):
    ax.plot([1, 2], [questionnaires.loc[i, 'NASA_vigilance1'], questionnaires.loc[i, 'NASA_vigilance2']], 'o-',
             color=colors[i], label=questionnaires.loc[i, 'task'] + ", S" + questionnaires.loc[i, 'subject_id'])
ax.set_ylabel('Vigilance rating', fontsize=18)
ax.set_ylim([0, 20])
ax.set_xticks([1, 2])
ax.set_xticklabels(['first half', 'second half'])
ax.tick_params(axis='both', which='major', labelsize=12)
plt.legend()
plt.show()

### Show drop in vigilance rating from first to second half
questionnaires['drop_vigilance_ratings'] = questionnaires['NASA_vigilance2'] - questionnaires['NASA_vigilance1']
fig, ax = plt.subplots()
for isubject in range(questionnaires.shape[0]):
    plt.text(0, questionnaires.loc[isubject, 'drop_vigilance_ratings'], questionnaires.loc[isubject, 'task'] + ", S" + questionnaires.loc[isubject, 'subject_id'], color=colors[isubject])
    plt.scatter(1+random.uniform(-0.3, 0.3), questionnaires.loc[isubject, 'drop_vigilance_ratings'], color=colors[isubject], label=questionnaires.loc[isubject, 'task'] + ", S" + questionnaires.loc[isubject, 'subject_id'])
plt.xlim([0, 1.75])
plt.legend()
plt.show()

print(questionnaires[['Task', 'Subject ID', 'drop_vigilance_ratings']])

### Vigilance drop by task

drops_vigilance_atc = list(questionnaires.loc[questionnaires.task=='atc', 'drop_vigilance_ratings'])
drops_vigilance_line_task_sim = list(questionnaires.loc[questionnaires.task=='line_task_sim', 'drop_vigilance_ratings'])
drops_vigilance_line_task_succ = list(questionnaires.loc[questionnaires.task=='line_task_succ', 'drop_vigilance_ratings'])
drops_vigilance_oddball = list(questionnaires.loc[questionnaires.task=='oddball', 'drop_vigilance_ratings'])
fix, ax = plt.subplots()
ax.scatter([1]*len(drops_vigilance_atc), drops_vigilance_atc)
ax.scatter([2]*len(drops_vigilance_line_task_sim), drops_vigilance_line_task_sim)
ax.scatter([3]*len(drops_vigilance_line_task_succ), drops_vigilance_line_task_succ)
ax.scatter([4]*len(drops_vigilance_oddball), drops_vigilance_oddball)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['atc', 'line_task_sim', 'line_task_succ', 'oddball'], rotation=45)
ax.set_title('Drop in vigilance first to second half of experiment')
plt.show()