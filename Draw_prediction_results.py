#%%
import matplotlib.pyplot as plt
import pickle


a = pickle.load(open('Results/eval-EXP-20201129-195018-model.pkl','rb'))

y_true = a['y_label']
y_pred = a['y_pred']
plt.rcParams['figure.figsize'] = [24.0, 4.0]

subject_range = range(len(y_true))
plt.figure()
plt.plot(subject_range, y_true, "ro", markersize=5, zorder=3, label=u"true_v")
plt.plot(subject_range, y_pred, "go", markersize=8, zorder=2, label=u"predict_v")
plt.legend(loc="upper left")
plt.xlim([-0.5, len(y_true)])
plt.xlabel("Subject")
plt.ylabel("Label")
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

# Draw Bar Chart for monos
plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (5, 3)


column_label = ['GlcNAc', 'Gal', 'End', 'Man', 'Fuc', 'Neu5Ac', 'GalNAc', 'Glc', 'Other']
occurrence = [1230, 1026, 600, 485, 283, 220, 208, 103, 32]
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


column_label = ['GlcNAc', 'Gal', 'End', 'Man', 'Fuc', 'Neu5Ac', 'GalNAc', 'Glc', 'Other']
occurrence = [1230, 1026, 600, 485, 283, 220, 208, 103, 32]
x = np.arange(len(column_label))

fig, ax = plt.subplots()
ax.bar(x, occurrence)
ax.set_xticks(x)
ax.set_xticklabels(column_label)
fig.autofmt_xdate()
plt.ylim([0, 1400])

plt.show()
