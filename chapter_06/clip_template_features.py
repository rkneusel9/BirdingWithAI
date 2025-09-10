#
#  file:  clip_template_features.py
#
#  Generate template CLIP embeddings
#
#  RTK, 16-Nov-2024
#  Last update:  16-Nov-2024
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
        im = preprocess(x).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(im)
        ans.append(features.cpu().numpy().squeeze())
    return np.array(ans)

def LoadImages():
    x = [Image.open("templates/chips/american_kestrel0_chip.png").convert("RGB"),
         Image.open("templates/chips/american_kestrel1_chip.png").convert("RGB"),
         Image.open("templates/chips/american_kestrel2_chip.png").convert("RGB"),
         Image.open("templates/chips/american_white_pelican0_chip.png").convert("RGB"),
         Image.open("templates/chips/american_white_pelican1_chip.png").convert("RGB"),
         Image.open("templates/chips/american_white_pelican2_chip.png").convert("RGB"),
         Image.open("templates/chips/belted_kingfisher0_chip.png").convert("RGB"),
         Image.open("templates/chips/belted_kingfisher1_chip.png").convert("RGB"),
         Image.open("templates/chips/belted_kingfisher2_chip.png").convert("RGB"),
         Image.open("templates/chips/great_blue_heron0_chip.png").convert("RGB"),
         Image.open("templates/chips/great_blue_heron1_chip.png").convert("RGB"),
         Image.open("templates/chips/great_blue_heron2_chip.png").convert("RGB"),
         Image.open("templates/chips/red-tailed_hawk0_chip.png").convert("RGB"),
         Image.open("templates/chips/red-tailed_hawk1_chip.png").convert("RGB"),
         Image.open("templates/chips/red-tailed_hawk2_chip.png").convert("RGB"),
         Image.open("templates/chips/snowy_egret0_chip.png").convert("RGB"),
         Image.open("templates/chips/snowy_egret1_chip.png").convert("RGB"),
         Image.open("templates/chips/snowy_egret2_chip.png").convert("RGB")]
    y = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5], dtype="uint8")
    return x,y

#  Configure the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Create template embeddings
x,y = LoadImages()
np.save("template_features.npy", CLIP(x, model, preprocess))
np.save("template_labels.npy", y)

