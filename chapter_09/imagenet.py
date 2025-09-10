#
#  file:  imagenet.py
#
#  Use Resnet-50 or MobileNetV3Large base models trained
#  on ImageNet for transfer learning
#
#  RTK, 23-Jan-2025
#  Last update:  03-Feb-2025
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


def ConfusionMatrix(pred, y, num_classes=10, per_class=False):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    pacc = np.zeros(num_classes)
    for i in range(cm.shape[0]):
        pacc[i] = cm[i,i] / cm[i].sum()
    return (cm, acc, pacc) if (per_class) else (cm, acc)


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
    xtrn = np.load("resnet/xtrain.npy")
    xtst = np.load("resnet/xtest.npy")
    ytrn = np.load("resnet/ytrain.npy")
    ytst = np.load("resnet/ytest.npy")
else:
    mode = "MobileNet features"
    xtrn = np.load("mobile/xtrain.npy")
    xtst = np.load("mobile/xtest.npy")
    ytrn = np.load("mobile/ytrain.npy")
    ytst = np.load("mobile/ytest.npy")

#  Define and train a top-level model
if (mtype == "RF"):
    clf = RandomForestClassifier(n_estimators=opt)
else:
    clf = MLPClassifier(hidden_layer_sizes=(opt,opt//2), max_iter=1000)
clf.fit(xtrn, ytrn)

#  Test it
pred = clf.predict(xtst)
cm, acc, pacc = ConfusionMatrix(pred, ytst, num_classes=10, per_class=True)
mcc = matthews_corrcoef(ytst, pred)

names = [
    "Black-bellied Whistling Duck", "Blue-gray Gnatcatcher",
    "Carolina Wren", "Limpkin", "Northern Cardinal",
    "Northern Mockingbird", "Northern Parula", 
    "Red-shouldered Hawk", "Sandhill Crane", "White-eyed Vireo"
]

print("%s %d, %s:" % (mtype, opt, mode))
for i in range(10):
    row = np.array2string(cm[i], separator=" ", formatter={'int': lambda x: f"{x:2d}"})
    print("%0.4f  %s  %s" % (pacc[i], row, names[i]))
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

