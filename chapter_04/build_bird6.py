#
#  file:  build_bird6.py
#
#  Build the bird6 datasets from the cropped 256x256 images
#
#  RTK, 28-Sep-2024
#  Last update:  28-Sep-2024
#
################################################################

import numpy as np
import os
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

    #  Get all the train and test filenames, scramble the order
    train_names = np.array(["train/"+i for i in os.listdir("train")])
    idx = np.argsort(np.random.random(len(train_names)))
    train_names = train_names[idx]

    test_names = np.array(["test/"+i for i in os.listdir("test")])
    idx = np.argsort(np.random.random(len(test_names)))
    test_names = test_names[idx]

    #  Use the filenames to assign class labels
    train_labels = np.array([ExtractLabel(i) for i in train_names])
    test_labels = np.array([ExtractLabel(i) for i in test_names])

    #  Train
    x64,x32,g64,g32 = ExtractImages(train_names)
    np.save("../data/bird6_64_xtrain.npy", x64)
    np.save("../data/bird6_32_xtrain.npy", x32)
    np.save("../data/bird6_gray_64_xtrain.npy", g64)
    np.save("../data/bird6_gray_32_xtrain.npy", g32)
    np.save("../data/bird6_ytrain.npy", train_labels)

    #  Test
    x64,x32,g64,g32 = ExtractImages(test_names)
    np.save("../data/bird6_64_xtest.npy", x64)
    np.save("../data/bird6_32_xtest.npy", x32)
    np.save("../data/bird6_gray_64_xtest.npy", g64)
    np.save("../data/bird6_gray_32_xtest.npy", g32)
    np.save("../data/bird6_ytest.npy", test_labels)

#  end build_birds6.py

