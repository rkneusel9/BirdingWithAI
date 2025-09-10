#
#  file:  nabirds_features.py
#
#  Generate NA birds features
#
#  RTK, 27-Nov-2024
#  Last update:  27-Nov-2024
#
################################################################

import time
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
        img = Image.fromarray(x)
        im = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(im)
        ans.append(features.cpu().numpy().squeeze())
    return np.array(ans)

#  Configure the model
print("Loading...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Create bird6 train and test image embeddings
print("Starting... ", end="", flush=True)
start = time.perf_counter()
x = np.load("nabirds_images.npy")
np.save("nabirds_features.npy", CLIP(x, model, preprocess))
end = time.perf_counter()
print("done... %0.2f seconds" % (end-start,), flush=True)

