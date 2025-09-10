#
#  file:  svm_image_generic.py
#
#  Use SVM trained on NA birds embeddings to classify an image
#
#  RTK, 02-Dec-2024
#  Last update: 02-Dec-2024
#
################################################################

import sys
import pickle
import torch
import clip
from sklearn.svm import SVC
import numpy as np
from PIL import Image

if (len(sys.argv) == 1):
    print()
    print("svm_image_generic <svm> <top-n> <image>")
    print()
    print("  <svm>   - trained SVM model name (.pkl)")
    print("  <top-n> - return top-n results")
    print("  <image> - the bird picture to classify")
    print()
    exit(0)

clf = pickle.load(open(sys.argv[1], "rb"))
top = int(sys.argv[2])
im = np.array(Image.open(sys.argv[3]).convert("RGB"))

#  Load the NA bird names
names = np.load("nabirds_names.npy")

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
image_features = features.cpu().numpy().squeeze().reshape((1,768))

#  Predict the class label using the decision function
#  scaled to [0,1]
prob = clf.decision_function(image_features).squeeze()
prob = (prob - prob.min()) / (prob.max() - prob.min())
idx = np.argsort(prob)[::-1][:top]

print("Image %s" % sys.argv[3])
for i in idx:
    print("  (%0.5f)  %s" % (prob[i], names[i]))
print()

