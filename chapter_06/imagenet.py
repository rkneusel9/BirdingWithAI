#
#  file:  imagenet.py
#
#  Use Resnet-50 or MobileNetV3Large base models trained
#  on ImageNet for transfer learning with bird6
#
#  RTK, 07-Nov-2024
#  Last update:  07-Nov-2024
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
    print("imagenet <top> <option> <model>")
    print()
    print("  <top>    - 'RF' or 'MLP'")
    print("  <option> - trees (RF) or hidden nodes (MLP)")
    print("  <model>  - 'resnet' or 'mobile'")
    print()
    exit(0)

mtype = sys.argv[1].upper()
opt = int(sys.argv[2])
mname = sys.argv[3].lower()

#  Load the appropriate embeddings
if (mname == "resnet"):
    mode = "ResNet-50 features"
    xtrn = np.load("results/features_resnet50/xtrain.npy")
    xtst = np.load("results/features_resnet50/xtest.npy")
    ytrn = np.load("results/features_resnet50/ytrain.npy")
    ytst = np.load("results/features_resnet50/ytest.npy")
else:
    mode = "MobileNet features"
    xtrn = np.load("results/features_mobilenet/xtrain.npy")
    xtst = np.load("results/features_mobilenet/xtest.npy")
    ytrn = np.load("results/features_mobilenet/ytrain.npy")
    ytst = np.load("results/features_mobilenet/ytest.npy")

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

