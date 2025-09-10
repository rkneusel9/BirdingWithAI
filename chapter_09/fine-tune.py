#
#  file:  fine-tune.py
#
#  RTK, 08-Feb-2025
#  Last update:  08-Feb-2025
#
################################################################

import os
import sys
import pickle
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


if (len(sys.argv) == 1):
    print()
    print("fine-tune <top> <option> <outdir>")
    print()
    print("  <top>    - 'RF' or 'MLP'")
    print("  <option> - trees (RF) or hidden nodes (MLP)")
    print("  <outdir> - output directory (overwritten)")
    print()
    exit(0)

mtype = sys.argv[1].upper()
opt = int(sys.argv[2])
outdir = sys.argv[3]

#  Load the appropriate embeddings
mode = "MobileNet fine-tune features"
xtrn = np.load("mobile_fine-tune_32_10/xtrain.npy")
xtst = np.load("mobile_fine-tune_32_10/xtest.npy")
ytrn = np.load("mobile_fine-tune_32_10/ytrain.npy")
ytst = np.load("mobile_fine-tune_32_10/ytest.npy")

#  Define and train a top-level model
if (mtype == "RF"):
    clf = RandomForestClassifier(n_estimators=opt)
else:
    clf = MLPClassifier(hidden_layer_sizes=(opt,opt//2), max_iter=1000)
clf.fit(xtrn, ytrn)

#  Test it
pred = clf.predict(xtst)
cm,acc = ConfusionMatrix(pred, ytst, num_classes=180)
mcc = matthews_corrcoef(ytst, pred)
print("%s %d, %s:" % (mtype, opt, mode))
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

#  Keep output
os.system("rm -rf %s 2>/dev/null" % outdir)
os.system("mkdir %s" % outdir)
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/predictions.npy", pred)
np.save(outdir+"/test_labels.npy", ytst)
pickle.dump(clf, open(outdir+("/%s.pkl" % mtype.lower()),"wb"))
with open(outdir+"/console.txt","w") as f:
    f.write("Test set accuracy: %0.4f, MCC: %0.4f\n" % (acc,mcc))

