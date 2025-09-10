#
#  file:  nabirds_name_embeddings.py
#
#  Calculate NA birds species common name embeddings
#
#  RTK, 27-Nov-2024
#  Last update:  27-Nov-2024
#
################################################################

import numpy as np
import torch
import clip

#  Common names
names = np.load("nabirds_names.npy")

#  Configure CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Embeddings
x = []
for p in names:
    text = clip.tokenize([p]).to(device)
    with torch.no_grad():
        f = model.encode_text(text).cpu().numpy()
    x.append(f)

x = np.array(x).squeeze()
np.save("nabirds_name_embeddings.npy", x)

