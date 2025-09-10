#
#  file:  imagenet_features.py
#
#  Use Resnet-50 or MobileNetV3Large base models trained
#  on ImageNet to generate features for transfer learning
#  with reference sonograms
#
#  RTK, 23-Jan-2025
#  Last update:  23-Jan-2025
#
################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pickle
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV3Large
from tensorflow.keras.applications.resnet50 import preprocess_input as rpreprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mpreprocess

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

if (len(sys.argv) == 1):
    print()
    print("imagenet_features <type> <outdir>")
    print()
    print("  <type>   - pretrained model type ('mobile' or 'resnet')")
    print("  <outdir> - output directory")
    print()
    exit(0)

mname = sys.argv[1].lower()
outdir = sys.argv[2]

#  Load the reference sonograms (train) and sample sonograms (test)
x = np.load("reference_sonograms.npy")
x_train = np.zeros((x.shape[0],336,336,3), dtype="uint8")
x_train[...,0] = x
x_train[...,1] = x
x_train[...,2] = x

x = np.load("samples_sonograms.npy")
x_test = np.zeros((x.shape[0],336,336,3), dtype="uint8")
x_test[...,0] = x
x_test[...,1] = x
x_test[...,2] = x

ytrain = np.load("reference_labels.npy")
ytest  = np.load("samples_labels.npy")

#  Augment training images
factor = 10
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

