#
#  file:  transfer.py
#
#  Experiments in transfer learning
#
#  RTK, 05-Nov-2024
#  Last update:  05-Nov-2024
#
################################################################

import os
import sys
import pickle
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

from vgg8 import VGG8
from resnet18 import ResNet18

from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Model, load_model


def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

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
    print("transfer <top> <option> <type> <factor>")
    print()
    print("  <top>    - 'RF' or 'MLP'")
    print("  <option> - trees (RF) or hidden nodes (MLP)")
    print("  <type>   - pretrained Birds 25 model type ('vgg8' or 'resnet18')")
    print("  <factor> - augmentation factor (e.g., 20)")
    print()
    exit(0)

mtype = sys.argv[1].lower()
opt = int(sys.argv[2])
mname = sys.argv[3].lower()
factor = int(sys.argv[4])

#  Load the bird6 train and test datasets (RGB, 64x64)
x_train = np.load("../data/bird6_64_xtrain.npy")
x_test  = np.load("../data/bird6_64_xtest.npy")
ytrain = np.load("../data/bird6_ytrain.npy")
ytest  = np.load("../data/bird6_ytest.npy")
input_shape = (64,64,3)
num_classes = 6

#  Augment training images
x_train, ytrain = AugmentDataset(x_train, ytrain, factor)

#  Scale [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#  Load the pretrained model and build ancillary model
if (mname == 'vgg8'):
    base = load_model('birds_25_vgg8.keras')
else:
    base = load_model('birds_25_resnet18.keras')

#  Build a new model to output the appropriate embedding
model = Model(inputs=base.input, outputs=base.layers[-3].output)

#  Pass the train and test data through the pretrained model
xtrn = model.predict(x_train, verbose=0)
xtst = model.predict(x_test, verbose=0)

#  Scale the embedding vectors to [0,1]
xtrn = (xtrn - xtrn.min()) / (xtrn.max() - xtrn.min())
xtst = (xtst - xtst.min()) / (xtst.max() - xtst.min())

#  Define and train a top-level model
if (mtype == "rf"):
    clf = RandomForestClassifier(n_estimators=opt)
else:
    clf = MLPClassifier(hidden_layer_sizes=(opt,opt//2), max_iter=1000)

clf.fit(xtrn, ytrain)

#  Test it using the embedding vectors
print("Embeddings (%s): (%s %d)" % (mname.upper(), mtype.upper(), opt))
pred = clf.predict(xtst)
cm,acc = ConfusionMatrix(pred, ytest, num_classes=num_classes)
mcc = matthews_corrcoef(ytest, pred)
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

#  Create a randomly initialized model
if (mname == "vgg8"):
    base = VGG8(input_shape, num_classes=num_classes)
else:
    base = ResNet18(input_shape, num_classes=num_classes)

model = Model(inputs=base.input, outputs=base.layers[-3].output)

xtrn = model.predict(x_train, verbose=0)
xtst = model.predict(x_test, verbose=0)
xtrn = (xtrn - xtrn.min()) / (xtrn.max() - xtrn.min())
xtst = (xtst - xtst.min()) / (xtst.max() - xtst.min())

if (mtype == "rf"):
    clf = RandomForestClassifier(n_estimators=opt)
else:
    clf = MLPClassifier(hidden_layer_sizes=(opt,opt//2), max_iter=1000)
clf.fit(xtrn, ytrain)

print("Random mapping:")
pred = clf.predict(xtst)
cm,acc = ConfusionMatrix(pred, ytest, num_classes=num_classes)
mcc = matthews_corrcoef(ytest, pred)
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

#  Finally, repeat using raw images unraveled
xxtrn = x_train.reshape((x_train.shape[0], 64*64*3))
xxtst = x_test.reshape((x_test.shape[0], 64*64*3))
xxtrn = (xxtrn - xxtrn.min()) / (xxtrn.max() - xxtrn.min())
xxtst = (xxtst - xxtst.min()) / (xxtst.max() - xxtst.min())

if (mtype == "rf"):
    clf = RandomForestClassifier(n_estimators=opt)
else:
    clf = MLPClassifier(hidden_layer_sizes=(opt,opt//2), max_iter=1000)
clf.fit(xxtrn, ytrain)

print("Unraveled images:")
pred = clf.predict(xxtst)
cm,acc = ConfusionMatrix(pred, ytest, num_classes=num_classes)
mcc = matthews_corrcoef(ytest, pred)
print(cm)
print("Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc))
print()

