#
#  file:  clip_template_classifier.py
#
#  Classify bird6 test CLIP features using template features
#
#  RTK, 16-Nov-2024
#  Last update: 16-Nov-2024
#
################################################################

import numpy as np
from sklearn.metrics import matthews_corrcoef

def Cosine(a,b):
    """Return the cosine distance between vectors a and b"""
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return 1.0 - np.dot(a,b) / (ma * mb)

def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

def Classify(x,y, xtst, ytst):
    """Use the templates to select a class label"""
    pred = []
    for i in range(len(xtst)):
        mn, lbl = 2,0
        for j in range(len(x)):
            sc = Cosine(x[j], xtst[i])
            if (sc < mn):
                mn,lbl = sc, y[j]
        pred.append(lbl)
    pred = np.array(pred)
    cm, acc = ConfusionMatrix(pred, ytst, num_classes=6)
    mcc = matthews_corrcoef(ytst, pred)
    return cm, acc, mcc


#  Load the template features and labels
x = np.load("template_features.npy")
y = np.load("template_labels.npy")

#  Load bird6 test embeddings
xtst = np.load("results/features_clip/bird6_clip_xtest.npy")
ytst = np.load("results/features_clip/bird6_clip_ytest.npy")

#  Using three templates per class
cm, acc, mcc = Classify(x, y, xtst, ytst)
print("3 templates per class:")
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

#  Using a single template per class
cm, acc, mcc = Classify(x[::3], y[::3], xtst, ytst)
print("1 template per class:")
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

#  Using the mean feature per class
xm = np.vstack((
    x[0:3].mean(axis=0), x[3:6].mean(axis=0), x[6:9].mean(axis=0),
    x[9:12].mean(axis=0), x[12:15].mean(axis=0), x[15:].mean(axis=0)))
cm, acc, mcc = Classify(xm, y[::3], xtst, ytst)
print("Mean template class:")
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

