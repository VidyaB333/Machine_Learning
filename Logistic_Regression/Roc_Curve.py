import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print(fpr)
print(tpr)
print(thresholds)
plt.plot(fpr, tpr)
plt.show()
print('Area under curve: ', metrics.auc(fpr, tpr))

print(metrics.roc_auc_score(y, scores))
