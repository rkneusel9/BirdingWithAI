#
#  file:  clip_detection.py
#
#  Use CLIP name embeddings for crude detection
#
#  RTK, 14-Dec-2024
#  Last update: 17-Dec-2024
#
################################################################

import os
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

def Chips(image, nr, nc):
    """Chip the image"""
    #  Rescale so that each chip is exactly 336x336
    #  user must enter relevant rows and cols closely matching
    #  the original image aspect ratio
    hnew, wnew = nr*336, nc*336
    img = Image.fromarray(image).resize((wnew,hnew), resample=Image.BILINEAR)
    img = np.array(img)
    chips = []
    for i in range(nr):
        for j in range(nc):
            chip = img[(i*336):(i*336+336), (j*336):(j*336+336), :]
            chips.append(chip)
    for i in range(nr-1):
        for j in range(nc-1):
            chip = img[(168+i*336):(168+i*336+336), (168+j*336):(168+j*336+336), :]
            chips.append(chip)
    return img, chips


if (len(sys.argv) == 1):
    print()
    print("clip_detection <threshold> <row> <col> <image> <outdir>")
    print()
    print("  <threshold> - keep top-1 if below threshold (per chip)")
    print("  <rows>, <cols> - chips (rows, cols)")
    print("  <image>     - the bird picture to classify")
    print("  <outdir>    - output directory")
    print()
    exit(0)

threshold = float(sys.argv[1])
nr,nc = int(sys.argv[2]), int(sys.argv[3])
image = np.array(Image.open(sys.argv[4]).convert("RGB"))
outdir = sys.argv[5]

#  Load the embeddings
num_classes = 404
names= np.load("nabirds_names.npy")
x = np.load("nabirds_name_embeddings.npy")
y = np.arange(num_classes, dtype="uint16")

#  Configure CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Chip the image
simage, chips = Chips(image, nr, nc)

#  Process
res = []
for chip in chips:
    #  Get the image embedding
    img = Image.fromarray(chip)
    im = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(im)
    image_features = features.cpu().numpy().squeeze()

    #  Find the top-1 best image embedding matches
    scores = []
    for i in range(len(x)):
        scores.append(Cosine(x[i], image_features))
    scores = np.array(scores)
    order = np.argsort(scores)
    score = scores[order][0]
    label = y[order][0]
    res.append((score, label))

os.system("rm -rf %s; mkdir %s" % (outdir,outdir))

timage = np.zeros(simage.shape, dtype="uint8")
gray = np.array(Image.fromarray(simage).convert("L"))
for i in range(3):
    timage[:,:,i] = gray

k, alpha = 0, 0.3333
shades = np.linspace(150,250,len(res))
for i in range(nr):
    for j in range(nc):
        score, label = res[k]
        if (score < threshold):
            a = gray[(i*336):(i*336+336), (j*336):(j*336+336)]
            b = shades[k]*np.ones((336,336), dtype="uint8")
            c = ((1-alpha)*a + alpha*b).astype("uint8")
            d = ((1-alpha)*a).astype("uint8")
            timage[(i*336):(i*336+336), (j*336):(j*336+336),0] = c
            timage[(i*336):(i*336+336), (j*336):(j*336+336),1] = d
            timage[(i*336):(i*336+336), (j*336):(j*336+336),2] = d
        k += 1
for i in range(nr-1):
    for j in range(nc-1):
        score, label = res[k]
        if (score < threshold):
            a = gray[(168+i*336):(168+i*336+336), (168+j*336):(168+j*336+336)]
            b = shades[k]*np.ones((336,336), dtype="uint8")
            c = ((1-alpha)*a + alpha*b).astype("uint8")
            d = ((1-alpha)*a).astype("uint8")
            timage[(168+i*336):(168+i*336+336), (168+j*336):(168+j*336+336),0] = c
            timage[(168+i*336):(168+i*336+336), (168+j*336):(168+j*336+336),1] = d
            timage[(168+i*336):(168+i*336+336), (168+j*336):(168+j*336+336),2] = d
        k += 1

Image.fromarray(image).save(outdir+"/original_image.png")
Image.fromarray(simage).save(outdir+"/scaled_image.png")
Image.fromarray(timage).save(outdir+"/overlay_image.png")

s = "Chip classification (%d rows, %d columns):\n" % (nr,nc)
k = 0
for i in range(nr):
    for j in range(nc):
        score, label = res[k]
        if (score < threshold):
            s += "%3d: (%0.5f)  %s\n" % (k, score, names[label])
        k += 1
for i in range(nr-1):
    for j in range(nc-1):
        score, label = res[k]
        if (score < threshold):
            s += "%3d: (%0.5f)  %s\n" % (k, score, names[label])
        k += 1

with open(outdir+"/console.txt", "w") as f:
    f.write(s)
print(s)

with open(outdir+"/command_line.txt", "w") as f:
    t = " ".join(sys.argv)
    f.write("%s\n" % t)

#  end clip_detection.py

