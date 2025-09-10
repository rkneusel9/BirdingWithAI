#
#  file: lenet_analysis.py
#
#  Evaluate the LeNet-style model
#
#  RTK, 08-Feb-2025
#  Last update:  10-Feb-2025
#
################################################################

import numpy as np
import matplotlib.pylab as plt

cm = np.load("results/lenet_32_9_14000/confusion_matrix.npy")
acc = np.diag(cm).sum() / cm.sum()
names = np.load("common_names.npy")

print("Overall accuracy: %0.4f" % acc)
print()
pacc = np.array([cm[i,i]/cm[i].sum() for i in range(180)])
order = np.argsort(pacc)[::-1]
for i in order[:10]:
    print("%0.4f  %s" % (pacc[i], names[i]))
print()
print("Mean accuracy %0.4f, median %0.4f" % (pacc.mean(), np.median(pacc)))
print()
print("%d species had an accuracy >0.5" % len(np.where(pacc>0.5)[0]))
print("%d species had an accuracy of 0" % len(np.where(pacc==0)[0]))
print()

x,y = range(180), pacc[order]
plt.plot(x,y, marker='none', color='k', linewidth=0.7)
plt.xlabel("Class (sorted)")
plt.ylabel("Accuracy (fraction)")
plt.tight_layout(pad=0.25, h_pad=0, w_pad=0)
plt.savefig("lenet_analysis_plot.png", dpi=300)
plt.close()
np.save("lenet_analysis_plot.npy", y)

