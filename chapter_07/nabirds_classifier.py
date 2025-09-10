#
#  file:  nabirds_classifier.py
#
#  Classify bird6, bird6 large, and NA birds features
#
#  RTK, 28-Nov-2024
#  Last update: 28-Nov-2024
#
################################################################

import os
import sys
import numpy as np
from sklearn.metrics import matthews_corrcoef

def Cosine(a,b):
    """Return the cosine distance between vectors a and b"""
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return 1.0 - np.dot(a,b) / (ma * mb)

def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    correct = []
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
        correct.append(1 if (pred[i]==y[i]) else 0)
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc, np.array(correct)

def Classify(x,y, xtst, ytst, num_classes=6):
    """Use the text embeddings to select a class label"""
    pred, dist = [],[]
    for i in range(len(xtst)):
        mn, lbl = 2,0
        for j in range(len(x)):
            sc = Cosine(x[j], xtst[i])
            if (sc < mn):
                mn,lbl = sc, y[j]
        pred.append(lbl)
        dist.append(mn)
    pred, dist = np.array(pred), np.array(dist)
    cm, acc, correct = ConfusionMatrix(pred, ytst, num_classes=num_classes)
    mcc = matthews_corrcoef(ytst, pred)
    return cm, acc, mcc, pred, dist, correct


if (len(sys.argv) == 1):
    print()
    print("nabirds_classifier <source> <outdir>")
    print()
    print("  <source> - 'bird6', 'large', or 'nabirds' features")
    print("  <outdir> - output directory (overwritten)")
    print()
    print("(N.B. bird6 uses nabirds name embeddings)")
    print()
    exit(0)

source, outdir = sys.argv[1:3]

#  Load the test set features
if (source == "bird6"):
    xtst = np.load("../chapter_07/results/features_clip/bird6_clip_xtest.npy")
    ytst = np.load("../chapter_07/results/features_clip/bird6_clip_ytest.npy")
elif (source == "large"):
    xtst = np.load("bird6_large_features.npy")
    ytst = np.load("bird6_large_labels.npy")
else:
    xtst = np.load("nabirds_features.npy")
    ytst = np.load("nabirds_labels.npy")

#  Recode ytst to use corresponding nabirds labels
if (source != 'nabirds'):
    i = np.where(ytst==0)[0]; ytst[i] =   9  # kestrel
    i = np.where(ytst==1)[0]; ytst[i] =  15  # pelican
    i = np.where(ytst==2)[0]; ytst[i] =  31  # kingfisher
    i = np.where(ytst==3)[0]; ytst[i] = 173  # blue heron
    i = np.where(ytst==4)[0]; ytst[i] = 296  # red-tail
    i = np.where(ytst==5)[0]; ytst[i] = 331  # snowy

num_classes = 404
names= np.load("nabirds_names.npy")
x = np.load("nabirds_name_embeddings.npy")
y = np.arange(num_classes, dtype="uint16")

#  Classify
cm, acc, mcc, pred, dist, correct = Classify(x, y, xtst, ytst, num_classes=num_classes)

#  Results
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))

s  = "Common name (%s):\n" % source
s += np.array2string(cm) + "\n"
s += "Test set accuracy: %0.4f, MCC: %0.4f\n\n" % (acc,mcc)

top = 5
t = ""
for i in range(num_classes):
    if (i in ytst):
        ac = 0.0 if (cm[i,i]==0) else cm[i,i]/cm[i].sum()
        t += "%-24s (%0.3f): " % (names[i][:24], ac)
        idx = np.argsort(cm[i])[::-1][:top]
        row = cm[i][idx]
        for j in range(len(idx)):
            if (cm[i,idx[j]] != 0):
                t += "%-20s (%3d)  " % (names[idx[j]][:20], row[j])
        t += "\n"

#  Display everything and store output
print("%s%s" % (s,t))
with open(outdir+"/console.txt", "w") as f:
    f.write(s)
    f.write(t)
with open(outdir+"/per_class.txt", "w") as f:
    f.write(t)
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/predictions.npy", pred)
np.save(outdir+"/distances.npy", dist)
np.save(outdir+"/correct.npy", correct)
np.save(outdir+"/labels.npy", ytst)

