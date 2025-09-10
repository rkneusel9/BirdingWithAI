#
#  file:  sparrow_features.py
#
#  Generate sparrow CLIP embeddings
#
#  RTK, 17-Nov-2024
#  Last update:  17-Nov-2024
#
################################################################

import os
import sys
import torch
import clip
import numpy as np
from PIL import Image

def Resize(x):
    """Make the image 336x336 pixels"""
    if (x.max() == 1.0):
        im = Image.fromarray((255*x).astype("uint8").convert("RGB"))
    else:

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

x = np.load("sparrow_images.npy")
np.save("sparrow_features.npy", CLIP(x, model, preprocess))

