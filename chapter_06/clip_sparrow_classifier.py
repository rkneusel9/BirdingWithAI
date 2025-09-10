#
#  file:  clip_sparrow_classifier.py
#
#  Classify sparrow images
#
#  RTK, 17-Nov-2024
#  Last update: 17-Nov-2024
#
################################################################

import numpy as np
from sklearn.metrics import matthews_corrcoef
import torch
import clip

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
    """Use the text embeddings to select a class label"""
    pred = []
    for i in range(len(xtst)):
        mn, lbl = 2,0
        for j in range(len(x)):
            sc = Cosine(x[j], xtst[i])
            if (sc < mn):
                mn,lbl = sc, y[j]
        pred.append(lbl)
    pred = np.array(pred)
    cm, acc = ConfusionMatrix(pred, ytst, num_classes=9)
    mcc = matthews_corrcoef(ytst, pred)
    return cm, acc, mcc

#  Configure CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Load the sparrow image embeddings
xtst = np.load("sparrow_features.npy")
ytst = np.load("sparrow_labels.npy")

prompts = [
    "Brewer's Sparrow", "Chipping Sparrow", "House Sparrow",
    "Lark Sparrow", "Lincoln's Sparrow", "Savannah Sparrow",
    "Song Sparrow", "Vesper Sparrow", "White-crowned Sparrow"
]

x = []
for p in prompts:
    text = clip.tokenize([p]).to(device)
    with torch.no_grad():
        f = model.encode_text(text).cpu().numpy()
    x.append(f)
x = np.array(x).squeeze()
y = np.array([0,1,2,3,4,5,6,7,8], dtype="uint8")

cm, acc, mcc = Classify(x, y, xtst, ytst)
print("Sparrows by common name:")
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

