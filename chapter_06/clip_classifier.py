#
#  file:  clip_classifier.py
#
#  Use CLIP features to train simple models
#
#  RTK, 14-Nov-2024
#  Last update:  14-Nov-2024
#
################################################################

import os
import sys
import numpy as np

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
    print("clip_classifier <top> <option>")
    print()
    print("  <top>    - 'RF' or 'MLP'")
    print("  <option> - trees (RF) or hidden nodes (MLP)")
    print()
    exit(0)

mtype = sys.argv[1].upper()
opt = int(sys.argv[2])

#  Load the appropriate embeddings
mode = "CLIP features"
xtrn = np.load("results/features_clip/bird6_clip_xtrain.npy")
xtst = np.load("results/features_clip/bird6_clip_xtest.npy")
ytrn = np.load("results/features_clip/bird6_clip_ytrain.npy")
ytst = np.load("results/features_clip/bird6_clip_ytest.npy")

#  Define and train a top-level model
if (mtype == "RF"):
    clf = RandomForestClassifier(n_estimators=opt)
else:
    clf = MLPClassifier(hidden_layer_sizes=(opt,opt//2), max_iter=1000)
clf.fit(xtrn, ytrn)

#  Test it
pred = clf.predict(xtst)
cm,acc = ConfusionMatrix(pred, ytst, num_classes=6)
mcc = matthews_corrcoef(ytst, pred)
print("%s %d, %s:" % (mtype, opt, mode))
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

