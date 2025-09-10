#
#  file: fine-tune_analysis.py
#
#  Evaluate the fine-tuned MobileNet model
#
#  RTK, 10-Feb-2025
#  Last update:  10-Feb-2025
#
################################################################

import numpy as np
import matplotlib.pylab as plt

cm = np.load("results/mobile_fine-tune_mlp_512/confusion_matrix.npy")
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

#  Compare to LeNet-style model
yl = np.load("lenet_analysis_plot.npy")

x,y = range(180), pacc[order]
plt.plot(x,y, marker='none', color='k', linewidth=0.7, label='MobileNet')
plt.plot(x,yl, marker='none', color='k', linewidth=0.7, linestyle='dashed', label='LeNet')
plt.xlabel("Class (sorted)")
plt.ylabel("Accuracy (fraction)")
plt.legend(loc='best')
plt.tight_layout(pad=0.25, h_pad=0, w_pad=0)
plt.savefig("fine-tune_analysis_plot.png", dpi=300)
plt.savefig("fine-tune_analysis_plot.eps", dpi=300)
plt.close()

