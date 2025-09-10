#
#  file:  clip_example.py
#
#  Directly comparing image and text CLIP embeddings
#
#  RTK, 20-Nov-2024
#  Last update: 20-Nov-2024
#
################################################################

import numpy as np
import torch
import clip
from PIL import Image

#  Configure CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Get the text embedding
text = "Belted Kingfisher"
tokens = clip.tokenize([text]).to(device)
with torch.no_grad():
    text = model.encode_text(tokens).cpu().numpy().squeeze()

#  And the image embeddings
image = Image.open("kingfisher.png").convert("RGB")
image = preprocess(image).unsqueeze(0).to(device)
with torch.no_grad():
    king = model.encode_image(image).cpu().numpy().squeeze()

image = Image.open("pelican.png").convert("RGB")
image = preprocess(image).unsqueeze(0).to(device)
with torch.no_grad():
    pelican = model.encode_image(image).cpu().numpy().squeeze()

import pdb; pdb.set_trace()

#  Calculate the cosine distances
mag_text = np.linalg.norm(text)
mag_king = np.linalg.norm(king)
mag_pelican = np.linalg.norm(pelican)

d_king = 1.0 - np.dot(king, text) / (mag_king * mag_text)
d_pelican = 1.0 - np.dot(pelican, text) / (mag_pelican * mag_text)

angle_king = np.arccos(1-d_king) * (180/np.pi)
angle_pelican = np.arccos(1-d_pelican) * (180/np.pi)

#  Compare
print("Magnitudes:")
print("  text      : %0.2f" % mag_text)
print("  kingfisher: %0.2f" % mag_king)
print("  pelican   : %0.2f" % mag_pelican)
print()
print("Cosine distance:")
print("  kingfisher: %0.4f" % d_king)
print("  pelican   : %0.4f" % d_pelican)
print()
print("Angles (degrees):")
print("  kingfisher: %0.2f" % angle_king)
print("  pelican   : %0.2f" % angle_pelican)
print()

