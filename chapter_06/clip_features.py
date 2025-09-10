#
#  file:  clip_features.py
#
#  Generate bird6 CLIP embeddings
#
#  N.B.
#     first run will cause PyTorch to download the CLIP model (891 MB)
#
#  RTK, 14-Nov-2024
#  Last update:  14-Nov-2024
#
################################################################

import os
import sys
import torch
import clip
import numpy as np
from PIL import Image

def CLIP(images, model, preprocess):
    """Return the image embeddings"""
    ans = []
    for x in images:
        img = Image.fromarray(x).convert("RGB")
        img = img.resize((336,336), resample=Image.BILINEAR)
        im = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(im)
        ans.append(features.cpu().numpy().squeeze())
    return np.array(ans)

if (len(sys.argv) == 1):
    print()
    print("clip_features <outdir>")
    print()
    print("  <outdir> - output directory")
    print()
    exit(0)

outdir = sys.argv[1]

#  Configure the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Create bird6 train and test image embeddings
os.system("rm -rf %s 2>/dev/null; mkdir %s" % (outdir,outdir))
x = np.load("../data/bird6_64_xtrain.npy")
np.save(outdir+"/bird6_clip_xtrain.npy", CLIP(x, model, preprocess))
x = np.load("../data/bird6_64_xtest.npy")
np.save(outdir+"/bird6_clip_xtest.npy", CLIP(x, model, preprocess))
x = np.load("../data/bird6_ytrain.npy")
np.save(outdir+"/bird6_clip_ytrain.npy", x)
x = np.load("../data/bird6_ytest.npy")
np.save(outdir+"/bird6_clip_ytest.npy", x)

