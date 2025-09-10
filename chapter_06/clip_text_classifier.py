#
#  file:  clip_text_classifier.py
#
#  Classify bird6 test CLIP features using text features
#
#  RTK, 17-Nov-2024
#  Last update: 26-Nov-2024
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
    cm, acc = ConfusionMatrix(pred, ytst, num_classes=6)
    mcc = matthews_corrcoef(ytst, pred)
    return cm, acc, mcc

#  Load the bird6 test set features
xtst = np.load("results/features_clip/bird6_clip_xtest.npy")
ytst = np.load("results/features_clip/bird6_clip_ytest.npy")

#  Configure CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Simple prompt embeddings
prompts = [
    "American Kestrel", "American White Pelican", "Belted Kingfisher",
    "Great Blue Heron", "Red-tailed Hawk", "Snowy Egret"
]

x = []
for p in prompts:
    text = clip.tokenize([p]).to(device)
    with torch.no_grad():
        f = model.encode_text(text).cpu().numpy()
    x.append(f)
x = np.array(x).squeeze()
y = np.array([0,1,2,3,4,5], dtype="uint8")

cm, acc, mcc = Classify(x, y, xtst, ytst)
print("Common name:")
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

#  Scientific name
prompts = [
    "Falco sparverius", "Pelecanus erythrorhynchos", "Megaceryle alcyon",
    "Ardea herodias", "Buteo jamaicensis", "Egretta thula"
]

x = []
for p in prompts:
    text = clip.tokenize([p]).to(device)
    with torch.no_grad():
        f = model.encode_text(text).cpu().numpy()
    x.append(f)
x = np.array(x).squeeze()
y = np.array([0,1,2,3,4,5], dtype="uint8")

cm, acc, mcc = Classify(x, y, xtst, ytst)
print("Scientific name:")
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

#  Bird description
prompts = [
"""
A small falcon with a striking plumage, it has a rusty-orange back and tail, a pale underside marked with dark spots, and slate-blue wings with black tips. Its head features bold black vertical stripes on a white face, offset by a gray crown and rusty nape. The tail shows a black band near the tip.
""",
"""
A massive waterbird with a snowy white body, long yellow-orange bill, and bright orange legs and feet. The wings are expansive with contrasting black flight feathers visible in flight. The bill features a distinctive pouch, and during breeding season, adults may display a small yellow crest and a horn-like ridge on the upper mandible.
""",
"""
A medium-sized bird with a slate-blue back, head, and wings, accented by a white underside and a blue breast band. Its shaggy crest gives it a distinctive silhouette, and females display an additional rusty band on the flanks. The long, pointed bill and short tail complement its bold, contrasting plumage.
""",
"""
A tall, elegant bird with slate-gray plumage, a long, sinuous neck, and a dagger-like yellow bill. Its head is adorned with a white face, a black eye stripe extending into long plumes, and a distinctive dark cap. The wings are broad and powerful, and the legs are long and dark, suited for wading.
""",
"""
A robust raptor with a pale, streaked underside and a warm, reddish-brown tail that contrasts sharply with its dark brown back and wings. The head is broad and dark with a hooked yellow-tipped bill, while its legs are feathered only partway down, ending in yellow talons.
""",
"""
A slender, pure white wading bird with a long, black bill and striking yellow lores near the eyes. Its black legs are offset by bright yellow feet. Delicate, wispy plumes adorn its back, neck, and crest, especially during breeding season, adding an elegant flair to its otherwise simple, sleek appearance.
"""
]

#  Text descriptions without the common name
x = []
for p in prompts:
    text = clip.tokenize([p]).to(device)
    with torch.no_grad():
        f = model.encode_text(text).cpu().numpy()
    x.append(f)
x = np.array(x).squeeze()
y = np.array([0,1,2,3,4,5], dtype="uint8")

cm, acc, mcc = Classify(x, y, xtst, ytst)
print("Text description:")
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

