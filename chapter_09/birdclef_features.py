#
#  file:  birdclef_features.py
#
#  Use Resnet-50 or MobileNetV3Large base models trained
#  on ImageNet to generate features for transfer learning
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

if (len(sys.argv) == 1):
    print()
    print("birdclef_features <type> <outdir>")
    print()
    print("  <type>   - pretrained model type ('mobile' or 'resnet')")
    print("  <outdir> - output directory")
    print()
    exit(0)

mname = sys.argv[1].lower()
outdir = sys.argv[2]

#  Load the train and test sonograms
x_train = np.load("birdclef_xtrain.npy")
ytrain  = np.load("birdclef_ytrain.npy")
x_test = np.load("birdclef_xtest.npy")
ytest  = np.load("birdclef_ytest.npy")

#  Load the proper pretrained model and set up proper preprocessing function
input_shape = (224, 224, 3)
if (mname == 'mobile'):
    model = MobileNetV3Large(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')
    preprocess = mpreprocess
else:
    model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')
    preprocess = rpreprocess 

#  Pass the train and test data through the pretrained model
xtrn = []
for i in range(len(x_train)):
    im = tf.image.resize(x_train[i], (224,224))
    im = preprocess(im)
    im = tf.expand_dims(im, axis=0)
    embedding = model.predict(im, verbose=0)
    xtrn.append(embedding.squeeze())
xtrn = np.array(xtrn, dtype="float32")

xtst = []
for i in range(len(x_test)):
    im = tf.image.resize(x_test[i], (224,224))
    im = preprocess(im)
    im = tf.expand_dims(im, axis=0)
    embedding = model.predict(im, verbose=0)
    xtst.append(embedding.squeeze())
xtst = np.array(xtst, dtype="float32")

#  Scale the embedding vectors to [0,1]
xtrn = (xtrn - xtrn.min()) / (xtrn.max() - xtrn.min())
xtst = (xtst - xtst.min()) / (xtst.max() - xtst.min())

#  And store on disk
os.system("rm -rf %s 2>/dev/null; mkdir %s" % (outdir,outdir))
np.save(outdir+"/xtrain.npy", xtrn)
np.save(outdir+"/ytrain.npy", ytrain)
np.save(outdir+"/xtest.npy", xtst)
np.save(outdir+"/ytest.npy", ytest)

