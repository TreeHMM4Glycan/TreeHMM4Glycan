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
