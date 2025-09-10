#
#  file:  build_birdclef.py
#
#  Create a BirdCLEF dataset using an 70/30 split
#
#  RTK, 23-Jan-2025
#  Last update:  23-Jan-2025
#
################################################################

import os
import random
import numpy as np
from PIL import Image

def Augment(im):
    """Augment by random rolling in time"""
    img = Image.fromarray(im)
    xs = np.random.randint(-20,21)
    im = np.roll(np.array(img), xs, axis=1)
    return im

def AugmentDataset(x,y, factor=10):
    """Augment the dataset"""
    if (x.ndim == 3):
        n, height, width = x.shape
        newx = np.zeros((n*factor, height, width), dtype="uint8")
    else:
        n, height, width, channels = x.shape
        newx = np.zeros((n*factor, height, width, channels), dtype="uint8")
    newy = np.zeros(n*factor, dtype="uint8")
    k=0 
    for i in range(n):
        im = Image.fromarray(x[i,:])
        newx[k,...] = np.array(im)
        newy[k] = y[i]
        k += 1
        for j in range(factor-1):
            newx[k,...] = Augment(x[i,:])
            newy[k] = y[i]
            k += 1
    idx = np.argsort(np.random.random(newx.shape[0]))
    return newx[idx], newy[idx]

random.seed(42)  # reproducible output
np.random.seed(42)

#  Get directory and common names
t = [i[:-1].split("|") for i in open("common_names.txt")]
dname = [i[0].strip() for i in t]
cname = [i[1].strip() for i in t]
np.save("common_names.npy", np.array(cname)) # preserve ordering

#  Gather train and test simultaneously
xtrn, ytrn = [],[]
xtst, ytst = [],[]

for i in range(len(dname)):
    f = [("BirdCLEF/train_images/%s/" % dname[i])+k for k in os.listdir("BirdCLEF/train_images/%s" % dname[i])]
    random.shuffle(f)
    n = int(0.7*len(f))
    ftrn, ftst = f[:n], f[n:]
    for t in ftrn:
        im = np.array(Image.open(t).convert("RGB"))
        xtrn.append(im)
        ytrn.append(i)
    for t in ftst:
        im = np.array(Image.open(t).convert("RGB"))
        xtst.append(im)
        ytst.append(i)

idx = np.argsort(np.random.random(len(xtrn)))
xtrn, ytrn = np.array(xtrn)[idx], np.array(ytrn)[idx]
idx = np.argsort(np.random.random(len(xtst)))
xtst, ytst = np.array(xtst)[idx], np.array(ytst)[idx]

#  Augment the training set
xtrn, ytrn = AugmentDataset(xtrn,ytrn, factor=3)

np.save("birdclef_xtrain.npy", xtrn)
np.save("birdclef_ytrain.npy", ytrn)
np.save("birdclef_xtest.npy", xtst)
np.save("birdclef_ytest.npy", ytst)

