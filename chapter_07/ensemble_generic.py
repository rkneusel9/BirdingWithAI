#
#  file:  ensemble_generic.py
#
#  Ensemble nearest-neighbor, MLP, and SVM classifiers
#
#  RTK, 12-Dec-2024
#  Last update: 12-Dec-2024
#
################################################################

import sys
import numpy as np
import pickle
import torch
import clip
from sklearn.neural_network import MLPClassifier
from PIL import Image

def Cosine(a,b):
    """Return the cosine distance between vectors a and b"""
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return 1.0 - np.dot(a,b) / (ma * mb)

if (len(sys.argv) == 1):
    print()
    print("ensemble_generic <mlp> <svm> <image>")
    print()
    print("  <mlp>    - name of the pretrained MLP")
    print("  <svm>    - name of the pretrained SVM")
    print("  <image>  - the bird picture to classify")
    print()
    exit(0)

mlp = pickle.load(open(sys.argv[1],"rb"))
svm = pickle.load(open(sys.argv[2],"rb"))
im = np.array(Image.open(sys.argv[3]).convert("RGB"))

#  Cosine distance:
num_classes = 404
names= np.load("nabirds_names.npy")
x = np.load("nabirds_averaged_features.npy")
y = np.arange(num_classes, dtype="uint16")

#  Configure CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Get the image embedding
h,w,_ = im.shape
z = max(h,w)
img = np.zeros((z,z,3), dtype="uint8")
xoff, yoff = (z-w)//2, (z-h)//2
img[yoff:(yoff+h), xoff:(xoff+w), :] = im
img = Image.fromarray(img).resize((336,336), resample=Image.BILINEAR)
im = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    features = model.encode_image(im)
image_features = features.cpu().numpy().squeeze()

#  cosine distance
scores = []
for i in range(len(x)):
    scores.append(Cosine(x[i], image_features))
scores = np.array(scores)
clabel = np.argsort(scores)[0]

#  MLP
prob = mlp.predict_proba(image_features.reshape((1,768))).squeeze()
mlabel = np.argsort(prob)[::-1][0]

#  SVM
prob = svm.decision_function(image_features.reshape((1,768))).squeeze()
prob = (prob - prob.min()) / (prob.max() - prob.min())
slabel = np.argsort(prob)[::-1][0]

#  Voting
if (clabel == mlabel) and (clabel == slabel):
    label = clabel  # all agree
elif (clabel == mlabel) and (clabel != slabel):
    label = clabel  # c and m agree
elif (mlabel == slabel) and (mlabel != clabel):
    label = mlabel  # m and s agree
elif (clabel == slabel) and (clabel != mlabel):
    label = clabel  # c and s agree
else:
    n = np.random.randint(0,3)  # select at random
    label = [clabel,mlabel,slabel][n]

print("Voting: %30s  (%s, %s, %s)" % (names[label], names[clabel], names[mlabel], names[slabel]))

