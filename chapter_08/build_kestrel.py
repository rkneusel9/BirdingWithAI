#
#  file:  build_kestrel.py
#
#  Build the kestrel dataset
#
#  RTK, 15-Dec-2024
#  Last update: 15-Dec-2024
#
################################################################

import os
import numpy as np
from PIL import Image

names = ["kestrel/"+i for i in os.listdir("kestrel")]
x1 = []
for name in names:
    img = np.array(Image.open(name).convert("RGB"))
    x1.append(img)
x1 = np.array(x1)

names = ["background/"+i for i in os.listdir("background")]
x0 = []
for name in names:
    img = np.array(Image.open(name).convert("RGB"))
    x0.append(img)
x0 = np.array(x0)

x = np.vstack((x0,x1))
y = np.array([0]*len(x0) + [1]*len(x1))
np.random.seed(6502)
i = np.argsort(np.random.random(len(y)))
x,y = x[i],y[i]
n = int(0.8*len(y))
xtrn,xtst = x[:n],x[n:]
ytrn,ytst = y[:n],y[n:]

np.save("data/kestrel_xtrain.npy", xtrn)
np.save("data/kestrel_ytrain.npy", ytrn)
np.save("data/kestrel_xtest.npy", xtst)
np.save("data/kestrel_ytest.npy", ytst)

