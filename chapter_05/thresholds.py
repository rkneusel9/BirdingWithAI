#
#  file:  thresholds.py
#
#  RTK, 23-Oct-2024
#  Last update:  23-Oct-2024
#
################################################################

import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from lenet5 import ConfusionMatrix

def Search(pc, pw):
    """Brute-force search for a threshold"""
    thresh, mx = 0.1, -1.0
    for t in np.linspace(0.1, 1.0, 10000):
        nc = len(np.where(pc >= t)[0])
        nw = len(np.where(pw < t)[0])
        score = nc/len(pc) + nw/len(pw)
        if (score > mx):
            mx = score
            thresh = t
    return thresh


if (len(sys.argv) == 1):
    print()
    print("thresholds rgb|gray 32|64 <model>")
    print()
    print("  rgb|gray - image type")
    print("  32|64    - image size")
    print("  <model>  - appropriate trained bird6 model")
    print()
    exit(0)

itype = sys.argv[1].lower()
isize = int(sys.argv[2])
mname = sys.argv[3]

#  Load the requested dataset
if (isize == 64):
    if (itype == "rgb"):
        xtest = np.load("../data/bird6_64_xtest.npy")
    else:
        xtest = np.load("../data/bird6_gray_64_xtest.npy")
else:
    if (itype == "rgb"):
        xtest = np.load("../data/bird6_32_xtest.npy")
    else:
        xtest = np.load("../data/bird6_gray_32_xtest.npy")

xtest = xtest / 255
ytest = np.load("../data/bird6_ytest.npy")

#  Load model
model = load_model(mname)

#  Split the test set in half -- one for threshold prediction
#  the other for testing the thresholds
n = len(ytest)//2
ytest0, ytest1 = ytest[:n], ytest[n:]
xtest0, xtest1 = xtest[:n], xtest[n:]

#  Get the model's predictions on samples used for thresholding
pred = model.predict(xtest0, verbose=0)
plabel = np.argmax(pred, axis=1)

#  Gather maximum softmax prediction for correct and incorrect predictions
pc = pred[np.where(plabel==ytest0)].max(axis=1)
pw = pred[np.where(plabel!=ytest0)].max(axis=1)

#  Search for a threshold
threshold = Search(pc,pw)

#  Get predictions on the remainder of the test set
pred1 = model.predict(xtest1, verbose=0)

#  Find predictions to label and ignore
ignore = np.where(pred1.max(axis=1) < threshold)[0]
keep = np.where(pred1.max(axis=1) >= threshold)[0]
ni,nk = len(ignore), len(keep)

#  Assign labels to the remaining predictions
l,y = np.argmax(pred1[keep], axis=1), ytest1[keep]
cm, acc = ConfusionMatrix(l,y, num_classes=6)

#  Report
print("Assigning labels to %d predictions (%0.5f) (threshold=%0.6f)" % (nk, nk/(ni+nk), threshold))
print(cm)
print("Overall accuracy = %0.5f" % acc)
print("Ignoring %d predictions (%0.5f)" % (ni, ni/(ni+nk)))
print()

