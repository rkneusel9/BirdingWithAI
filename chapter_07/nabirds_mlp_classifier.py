#
#  file:  nabirds_mlp_classifier.py
#
#  Classify bird6, bird6 large, and NA birds features
#  using an MLP trained on NA bird embeddings
#
#  RTK, 01-Dec-2024
#  Last update: 01-Dec-2024
#
################################################################

import os
import sys
import numpy as np
import pickle
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier

def TrainTestSplit(X,Y):
    """Split NA birds embeddings train and test"""
    np.random.seed(42)
    for c in range(404):
        x = X[np.where(y==c)]
        i = np.argsort(np.random.random(len(x)))
        x = x[i]
        n = int(0.625*len(x))
        if (c==0):
            xtrn, xtst = x[:n], x[n:]
            ytrn = np.zeros(len(xtrn), dtype="uint16")
            ytst = np.zeros(len(xtst), dtype="uint16")
        else:
            xtrn = np.vstack((xtrn, x[:n]))
            xtst = np.vstack((xtst, x[n:]))
            ytrn = np.hstack((ytrn, c*np.ones(len(x[:n]), dtype="uint16")))
            ytst = np.hstack((ytst, c*np.ones(len(x[n:]), dtype="uint16")))

    i = np.argsort(np.random.random(len(xtrn)))
    xtrn, ytrn = xtrn[i], ytrn[i]
    i = np.argsort(np.random.random(len(xtst)))
    xtst, ytst = xtst[i], ytst[i]
    np.random.seed()
    return xtrn,ytrn, xtst,ytst

def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    correct = []
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
        correct.append(1 if (pred[i]==y[i]) else 0)
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc, np.array(correct)


if (len(sys.argv) == 1):
    print()
    print("nabirds_mlp_classifier <nodes> <source> <outdir>")
    print()
    print("  <nodes>  - first hidden layer nodes (e.g. 512)")
    print("  <source> - 'bird6', 'large', or 'nabirds' features")
    print("  <outdir> - output directory (overwritten)")
    print()
    print("(N.B. bird6 uses nabirds name embeddings)")
    print()
    exit(0)

nodes  = int(sys.argv[1])
source = sys.argv[2]
outdir = sys.argv[3]

#  Load the test set features
x = np.load("nabirds_features.npy")
y = np.load("nabirds_labels.npy")

xtrn,ytrn, xtst,ytst = TrainTestSplit(x,y)

if (source == "bird6"):
    xtst = np.load("../chapter_07/results/features_clip/bird6_clip_xtest.npy")
    ytst = np.load("../chapter_07/results/features_clip/bird6_clip_ytest.npy")
elif (source == "large"):
    xtst = np.load("bird6_large_features.npy")
    ytst = np.load("bird6_large_labels.npy")

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

#  Define and train the MLP
print("Training... ", end="", flush=True)
clf = MLPClassifier(hidden_layer_sizes=(nodes, nodes//2), max_iter=1000)
clf.fit(xtrn, ytrn)
print("complete", flush=True)
print()

#  Classify
pred = clf.predict(xtst)
cm, acc, correct = ConfusionMatrix(pred, ytst, num_classes=num_classes)
mcc = matthews_corrcoef(ytst, pred)

#  Results
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
s  = "MLP %d (%s):\n" % (nodes, source)
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
np.save(outdir+"/correct.npy", correct)
np.save(outdir+"/labels.npy", ytst)
pickle.dump(clf, open(outdir+"/mlp.pkl", "wb"))

