#
#  file:  nabirds_classifier_results.py
#
#  Interpret the output of nabirds_classifier.py
#  and nabirds_mlp_classifier.py
#
#  RTK, 30-Nov-2024
#  Last update:  01-Dec-2024
#
################################################################

import os
import sys
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import ttest_ind, mannwhitneyu

def Cohens(a,b):
    """Calculate Cohen's d"""
    sa,sb = a.std(ddof=1), b.std(ddof=1)
    na,nb = len(a), len(b) 
    pool = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    return (a.mean() - b.mean()) / pool

def MAD(x):
    """Median absolute deviation"""
    return np.median(np.abs(x-np.median(x)))

if (len(sys.argv) == 1):
    print()
    print("nabirds_classifier_results <srcdir>")
    print()
    print("  <srcdir> - directory containing output files")
    print()
    exit(0)

base = sys.argv[1]

#  Load results files
cm   = np.load(base+"/confusion_matrix.npy")
nc   = np.load(base+"/correct.npy")
try:
    dist = np.load(base+"/distances.npy")
except:
    dist = None
pred = np.load(base+"/predictions.npy")
y    = np.load(base+"/labels.npy")

#  Compare distances for correct and incorrect
if (dist is not None):
    d0,d1 = dist[np.where(nc==0)], dist[np.where(nc==1)]
    print("Distance, correct vs incorrect:")
    print("    correct  : %0.6f +/- %0.6f" % (d1.mean(), d1.std(ddof=1)/np.sqrt(len(d1))))
    print("    incorrect: %0.6f +/- %0.6f" % (d0.mean(), d0.std(ddof=1)/np.sqrt(len(d0))))
    t,p = ttest_ind(d0,d1)
    _,u = mannwhitneyu(d0,d1)
    d = Cohens(d0,d1)
    print()
    print("(t=% 0.8f, p=%0.8f, u=%0.8f, Cohen's d=%2.3f)" % (t,p,u,d))
    print()

#  Distribution of accuracies
acc = []
for i in range(cm.shape[0]):
    if (cm[i].sum() != 0):
        acc.append(cm[i,i] / cm[i].sum())
acc = np.array(acc)
print("Accuracy:")
print("    overall       : %0.5f" % (np.diag(cm).sum() / cm.sum(),))   
print("    mean per class: %0.5f +/- %0.5f (SD)" % (acc.mean(), acc.std(ddof=1)))
print("    median        : %0.5f +/- %0.5f (MAD)" % (np.median(acc), MAD(acc)))
print()

#  Accuracy histograms
h,x = np.histogram(acc, bins=10)
h,x = h/h.sum(), 0.5*(x[1:]+x[:-1])
plt.bar(x,h, fill=False, edgecolor='k', width=0.9*(x[1]-x[0]))
plt.xlabel("accuracy (per class)")
plt.ylabel("fraction")
plt.tight_layout(pad=0.25, h_pad=0, w_pad=0)
plt.savefig("nabirds_results_acc_plot.png", dpi=300)
plt.savefig("nabirds_results_acc_plot.eps", dpi=300)
plt.show()

#  Distance histograms
if (dist is not None):
    h0,x0 = np.histogram(d0, bins=100)
    h0,x0 = h0/h0.sum(), 0.5*(x0[1:]+x0[:-1])
    h1,x1 = np.histogram(d1, bins=100)
    h1,x1 = h1/h1.sum(), 0.5*(x1[1:]+x1[:-1])

    plt.bar(x0,h0, fill=True, edgecolor='b', width=0.9*(x0[1]-x0[0]), label="incorrect")
    plt.bar(x1,h1, fill=True, edgecolor='r', width=0.9*(x1[1]-x1[0]), label="correct")
    plt.xlabel("distance")
    plt.ylabel("fraction")
    plt.legend(loc='best')
    plt.tight_layout(pad=0.25, h_pad=0, w_pad=0)
    plt.savefig("nabirds_results_dist_plot.png", dpi=300)
    plt.savefig("nabirds_results_dist_plot.eps", dpi=300)
    plt.show()

