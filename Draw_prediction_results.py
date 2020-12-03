#%%
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

target_protein = 'AAL (100 ug/ml)'
target_result_file = 'Results/eval-EXP-20201129-195018-model.pkl'
a = pickle.load(open(target_result_file, 'rb'))

y_iupac = a['y_iupac']
y_true = a['y_label']
y_pred = a['y_pred']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [12.0, 4.0]

iupac_data = pd.read_csv('./Data/IUPAC.csv')
iupacs = iupac_data['IUPAC'].tolist()
mscore_data = pd.read_csv('./Data/MScore_useful.csv')
mscore = mscore_data[target_protein].tolist()

cv_order = sorted(list(zip(y_iupac, y_true, y_pred)))
excel_order = sorted(list(zip(iupacs, mscore)))

iupacs, y_true, y_pred = list(zip(*cv_order))
_, mscore = list(zip(*excel_order))

pack = sorted(list(zip(mscore, y_true, y_pred, iupacs)))
mscore, y_true, y_pred, iupacs = list(zip(*pack))

subject_range = range(len(y_true))
fig, ax1 = plt.subplots()
ax1.plot(subject_range, y_true, "ro", markersize=4, zorder=3, label="True Class")
ax1.plot(subject_range, y_pred, "go", markersize=8, zorder=2, label="Predict Class")
ax1.set_xlim([-0.5, len(y_true)])
ax1.set_xlabel("Proteins")
ax1.set_ylabel("Labels and Prediction Results")

ax2 = ax1.twinx()
ax2.plot(subject_range, mscore, label="M Score")
ax2.set_ylabel("M Score")
ax2.hlines(y=2.0, xmin=300, xmax=500, colors='purple', linestyles='-', lw=2, label='Threshold for Binary Classification')
# ax2.legend(loc="upper left")
# To combine two legend together
ax1.plot(np.nan, color='purple', label='Threshold for Binary Classification')
ax1.legend(loc="upper left")

plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

# Draw Bar Chart for monos
plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (5, 3)


column_label = ['GlcNAc', 'Gal', 'End', 'Man', 'Fuc', 'Neu5Ac', 'GalNAc', 'Glc', 'Others']
occurrence = [1230, 1026, 945, 485, 283, 220, 208, 103, 32]
x = np.arange(len(column_label))

fig, ax = plt.subplots()
ax.bar(x, occurrence)
ax.set_xticks(x)
ax.set_xticklabels(column_label)
fig.autofmt_xdate()
plt.ylim([0, 1400])

plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

# Draw Bar Chart for links
plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (5, 3)


column_label = [r'$(\beta 1-4)$', 'End-Linkage', 'Start-Linkage', r'$(\beta 1-3)$', r'$(\alpha 1-3)$', r'$(\beta 1-2)$', r'$(\beta 1-6)$', r'$(\beta 1-2)$', r'$(\beta 2-3)$', r'$(\beta 1-6)$', r'$(\beta 2-6)$', r'$(\beta 1-4)$', 'Others']
occurrence = [1086, 945, 600, 564, 332, 261, 196, 159, 129, 99, 76, 63, 32]
x = np.arange(len(column_label))

fig, ax = plt.subplots()
ax.bar(x, occurrence)
ax.set_xticks(x)
ax.set_xticklabels(column_label)
fig.autofmt_xdate()
plt.ylim([0, 1200])

plt.show()
