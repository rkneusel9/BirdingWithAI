#
#  file:  ensemble.py
#
#  Ensemble the given models
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

def Average(pred):
    """Average the predictions"""
    ans = pred[0]
    for i in range(1,len(pred)):
        ans += pred[i]
    return np.argmax(ans / len(pred), axis=1)

def Geometric(pred):
    """Geometric mean"""
    ans = pred[0]
    for i in range(1,len(pred)):
        ans *= pred[i]
    return np.argmax(ans**(1/len(pred)), axis=1)

def Vote(pred):
    """Majority vote"""
    ans = []
    for i in range(pred[0].shape[0]):
        v = []
        for j in range(len(pred)):
            v.append(np.argmax(pred[j][i]))
        w = np.argmax(np.bincount(v, minlength=6))
        ans.append(w)
    return np.array(ans)

if (len(sys.argv) == 1):
    print()
    print("ensemble <x> <y> avg|geo|vote <model1> <model2> [<model3> ...]")
    print()
    print("  <x>, <y> - test data appropriate for given models")
    print("  avg|geo|vote - ensemble mode")
    print("  <model1> ... - models to ensemble")
    print()
    exit(0)

x = np.load(sys.argv[1]) / 255
y = np.load(sys.argv[2])
mode = sys.argv[3].lower()
models = [load_model(i) for i in sys.argv[4:]]

#  Get each model's predictions for the test set
pred = []
for model in models:
    pred.append(model.predict(x,verbose=0))

#  Ensemble by type
if (mode == "avg"):
    plabel = Average(pred)
elif (mode == "geo"):
    plabel = Geometric(pred)
else:
    plabel = Vote(pred)

#  Report
cm,acc = ConfusionMatrix(plabel, y, num_classes=6)
print()
print(cm)
print("Overall accuracy: %0.5f" % acc)
print()

