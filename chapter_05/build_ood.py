#
#  file:  build_ood.py
#
#  Bundle the OOD images
#
#  RTK, 17-Oct-2024
#  Last update:  17-Oct-2024
#
################################################################

import numpy as np
import os
from PIL import Image

def ExtractLabel(name):
    """Extract a label from a filename"""
    if   ("cooper" in name): ans=0
    elif ("egret" in name):  ans=1
    elif ("swain" in name):  ans=2
    elif ("heron" in name):  ans=3
    elif ("says" in name):   ans=4
    elif ("dragon" in name): ans=5
    elif ("flower" in name): ans=6
    else:
        raise ValueError("Unknown class: %s" % name)
    return ans


def ExtractImages(names):
    """Build image datasets"""
    N = len(names)
    x64 = np.zeros((N,64,64,3), dtype="uint8")
    x32 = np.zeros((N,32,32,3), dtype="uint8")
    g64 = np.zeros((N,64,64), dtype="uint8")
    g32 = np.zeros((N,32,32), dtype="uint8")

    for i in range(N):
        img = Image.open(names[i]).convert("RGB")
        i64 = img.resize((64,64), resample=Image.BILINEAR)
        i32 = img.resize((32,32), resample=Image.BILINEAR)
        x64[i,...] = np.array(i64)
        x32[i,...] = np.array(i32)
        g64[i,...] = np.array(i64.convert("L"))
        g32[i,...] = np.array(i32.convert("L"))

    return x64,x32,g64,g32


if (__name__ == "__main__"):
    #  Set the pseudorandom number seed
    np.random.seed(19937)

    names = np.array(["chips/"+i for i in os.listdir("chips")])
    idx = np.argsort(np.random.random(len(names)))
    names = names[idx]
    labels = np.array([ExtractLabel(i) for i in names])
    x64,x32,g64,g32 = ExtractImages(names)
    np.save("../data/ood_64_xtrain.npy", x64)
    np.save("../data/ood_32_xtrain.npy", x32)
    np.save("../data/ood_gray_64_xtrain.npy", g64)
    np.save("../data/ood_gray_32_xtrain.npy", g32)
    np.save("../data/ood_ytrain.npy", labels)

#  end build_ood.py

