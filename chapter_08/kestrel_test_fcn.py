#
#  file:  kestrel_test_fcn.py
#
#  Test that the fully convolutional version of the
#  kestrel model still works with single images
#
#  RTK, 15-Dec-2024
#  Last update:  15-Dec-2024
#
################################################################

import os
import sys
import numpy as np
from sklearn.metrics import matthews_corrcoef
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

if (len(sys.argv) == 1):
    print()
    print("kestrel_test_fcn <fcn>")
    print()
    print("  <fcn> - fully convolutional kestrel model (.keras)")
    print()
    exit(0)

#  Load the data
num_classes = 2
xtst = np.load("data/kestrel_xtest.npy")/255
ytst = np.load("data/kestrel_ytest.npy")

#  Load the model and predict
model = load_model(sys.argv[1])
pred = model.predict(xtst, verbose=0)
plabel = np.argmax(pred.squeeze(), axis=1)
cm,acc = ConfusionMatrix(plabel, ytst, num_classes=2)
mcc = matthews_corrcoef(ytst, plabel)
print(cm)
print('Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc))
print()

