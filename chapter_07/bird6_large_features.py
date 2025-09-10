#
#  file:  bird6_large_features.py
#
#  Generate bird6 full size CLIP embeddings
#
#  RTK, 28-Nov-2024
#  Last update:  28-Nov-2024
#
################################################################

import os
import sys
import torch
import clip
import numpy as np
from PIL import Image

def ExtractLabel(name):
    """Extract a label from a filename"""
    if   ("kestrel" in name):  ans=0
    elif ("pelican" in name):  ans=1
    elif ("kingf" in name):    ans=2
    elif ("heron" in name):    ans=3
    elif ("red-tail" in name): ans=4
    elif ("snowy" in name):    ans=5
    else:
        raise ValueError("Unknown class: %s" % name)
    return ans

def CLIP(names, model, preprocess):
    """Return the image embeddings"""
    ans, y = [], []
    for name in names:
        im = np.array(Image.open(name).convert("RGB"))
        h,w,_ = im.shape
        z = max(h,w)
        img = np.zeros((z,z,3), dtype="uint8")
        xoff, yoff = (z-w)//2, (z-h)//2
        img[yoff:(yoff+h), xoff:(xoff+w), :] = im
        img = Image.fromarray(img).resize((336,336), resample=Image.BILINEAR)
        im = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(im)
        ans.append(features.cpu().numpy().squeeze())
        y.append(ExtractLabel(name))
    return np.array(ans), np.array(y)


#  File names of the full bird6 test images
names = [("../chapter_05/done/test/"+i).replace("_chip","") for i in os.listdir("../chapter_05/test/")]

#  Configure the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Create the embeddings
emb,y = CLIP(names, model, preprocess)
np.save("bird6_large_features.npy", emb)
np.save("bird6_large_labels.npy", y)

