#
#  file:  clip_image_generic.py
#
#  Use CLIP image embeddings for classification of an unknown
#  image (zero-shot classification)
#
#  RTK, 29-Nov-2024
#  Last update: 29-Nov-2024
#
################################################################

import sys
import torch
import clip
import numpy as np
from PIL import Image

def Cosine(a,b):
    """Return the cosine distance between vectors a and b"""
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return 1.0 - np.dot(a,b) / (ma * mb)

if (len(sys.argv) == 1):
    print()
    print("clip_image_generic <top-n> <mode> <image> [<negate>]")
    print()
    print("  <top-n>  - capture the top-n best matches")
    print("  <mode>   - 'all', 'avg', 'navg', or 'name'")
    print("  <image>  - the bird picture to classify")
    print("  <negate> - 'negate' to keep the top-n _worst_ matches")
    print()
    exit(0)

topn = int(sys.argv[1])
mode = sys.argv[2]
im = np.array(Image.open(sys.argv[3]).convert("RGB"))
negate = (len(sys.argv) == 5)

#  Load the proper embeddings to scan
num_classes = 404
names= np.load("nabirds_names.npy")

if (mode == "name"):
    x = np.load("nabirds_name_embeddings.npy")
    y = np.arange(num_classes, dtype="uint16")
elif (mode == "all"):
    x = np.load("nabirds_features.npy")
    y = np.load("nabirds_labels.npy")
elif (mode == "avg"):
    x = np.load("nabirds_averaged_features.npy")
    y = np.arange(num_classes, dtype="uint16")
elif (mode == "navg"):
    x = np.load("nabirds_averaged_name_features.npy")
    y = np.arange(num_classes, dtype="uint16")
else:
    raise ValueError("unknown mode: %s" % mode)

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

#  Find the top-n best or worst image embedding matches
#  using cosine distance
scores = []
for i in range(len(x)):
    scores.append(Cosine(x[i], image_features))
scores = np.array(scores)

#  Sort everything together to extract scores, labels, and images
#  keeping the top-n smallest cosine distances
if (negate):
    order = np.argsort(scores)[::-1][:topn]
else:
    order = np.argsort(scores)[:topn]
scores = scores[order][:topn]
y = y[order][:topn]

#  Dump the results
print("Image %s (%s)" % (sys.argv[3], mode))
print()
for i in range(len(scores)):
    print("(%0.6f)  %s" % (scores[i], names[y[i]]))
print()

