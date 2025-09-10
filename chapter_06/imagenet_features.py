#
#  file:  imagenet_features.py
#
#  Use Resnet-50 or MobileNetV3Large base models trained
#  on ImageNet to generate features for transfer learning
#
#  RTK, 07-Nov-2024
#  Last update:  07-Nov-2024
#
################################################################

import os
import sys
import pickle
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV3Large
from tensorflow.keras.applications.resnet50 import preprocess_input as rpreprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mpreprocess

def Augment(im):
    img = Image.fromarray(im)
    if (np.random.random() < 0.5):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if (np.random.random() < 0.5):
        r = 3*np.random.random()-3
        img = img.rotate(r, resample=Image.BILINEAR)
    if (np.random.random() < 0.5):
        i = np.array(img)
        n = int(0.1*i.shape[0])
        i = np.roll(i, np.random.randint(-n,n+1), axis=1)
        i = np.roll(i, np.random.randint(-n,n+1), axis=0)
        img = Image.fromarray(i)
    return np.array(img)

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


if (len(sys.argv) == 1):
    print()
    print("imagenet_features <type> <factor> <outdir>")
    print()
    print("  <type>   - pretrained model type ('mobile' or 'resnet')")
    print("  <factor> - augmentation factor (e.g., 20)")
    print("  <outdir> - output directory")
    print()
    exit(0)

mname = sys.argv[1].lower()
factor = int(sys.argv[2])
outdir = sys.argv[3]

#  Load the bird6 train and test datasets (RGB, 64x64)
x_train = np.load("../data/bird6_64_xtrain.npy")
x_test  = np.load("../data/bird6_64_xtest.npy")
ytrain = np.load("../data/bird6_ytrain.npy")
ytest  = np.load("../data/bird6_ytest.npy")

#  Augment training images
x_train, ytrain = AugmentDataset(x_train, ytrain, factor)

#  Load the proper pretrained model and set up proper preprocessing function
if (mname == 'mobile'):
    model = MobileNetV3Large(weights='imagenet', include_top=False, pooling='avg')
    preprocess = mpreprocess
else:
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    preprocess = rpreprocess 

#  Pass the train and test data through the pretrained model
xtrn = []
for i in range(len(x_train)):
    im = tf.image.resize(x_train[i], (224,224))
    im = preprocess(im)
    im = tf.expand_dims(im, axis=0)
    embedding = model.predict(im, verbose=0)
    xtrn.append(embedding.squeeze())
xtrn = np.array(xtrn)

xtst = []
for i in range(len(x_test)):
    im = tf.image.resize(x_test[i], (224,224))
    im = preprocess(im)
    im = tf.expand_dims(im, axis=0)
    embedding = model.predict(im, verbose=0)
    xtst.append(embedding.squeeze())
xtst = np.array(xtst)

#  Scale the embedding vectors to [0,1]
xtrn = (xtrn - xtrn.min()) / (xtrn.max() - xtrn.min())
xtst = (xtst - xtst.min()) / (xtst.max() - xtst.min())

#  And store on disk
os.system("rm -rf %s 2>/dev/null; mkdir %s" % (outdir,outdir))
np.save(outdir+"/xtrain.npy", xtrn)
np.save(outdir+"/ytrain.npy", ytrain)
np.save(outdir+"/xtest.npy", xtst)
np.save(outdir+"/ytest.npy", ytest)

