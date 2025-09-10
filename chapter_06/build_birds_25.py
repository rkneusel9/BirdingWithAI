#
#  file:  build_birds_25.py
#
#  Build a 64x64 RGB dataset from the Birds 25 images
#
#  RTK, 03-Nov-2024
#  Last update:  03-Nov-2024
#
################################################################

import os
import numpy as np
from PIL import Image

def Chip(fname):
    img = np.array(Image.open(fname).convert("RGB"))
    h,w = img.shape[:2]
    if (h > w):
        lo,hi = (h-w)//2, (h-w)//2 + w
        im = img[lo:hi,:,:]
    else:
        lo,hi = (w-h)//2, (w-h)//2 + h
        im = img[:,lo:hi,:]
    img = Image.fromarray(im).resize((64,64), Image.BILINEAR)
    return np.array(img)

classes = sorted([i for i in os.listdir("Birds_25/train")])

xtrain, ytrain = [], []
for c,cl in enumerate(classes):
    files = ["Birds_25/train/"+cl+"/"+i for i in os.listdir("Birds_25/train/"+cl)]
    for f in files:
        img = Chip(f)
        xtrain.append(img)
        ytrain.append(c)
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
idx = np.argsort(np.random.random(len(ytrain)))
xtrain, ytrain = xtrain[idx], ytrain[idx]
np.save("../src/data/birds_25_xtrain.npy", xtrain)
np.save("../src/data/birds_25_ytrain.npy", ytrain)

xtest, ytest = [], []
for c,cl in enumerate(classes):
    files = ["Birds_25/valid/"+cl+"/"+i for i in os.listdir("Birds_25/valid/"+cl)]
    for f in files:
        img = Chip(f)
        xtest.append(img)
        ytest.append(c)
xtest = np.array(xtest)
ytest = np.array(ytest)
idx = np.argsort(np.random.random(len(ytest)))
xtest, ytest = xtest[idx], ytest[idx]
np.save("../src/data/birds_25_xtest.npy", xtest)
np.save("../src/data/birds_25_ytest.npy", ytest)

