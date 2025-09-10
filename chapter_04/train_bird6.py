#
#  file:  train_bird6.py
#
#  Train bird6 models
#
#  RTK, 28-Sep-2024
#  Last update:  28-Sep-2024
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
import numpy as np
import matplotlib.pylab as plt

from lenet5   import LeNet5, ConfusionMatrix
from vgg8     import VGG8
from resnet18 import ResNet18

#  Command line
if (len(sys.argv) == 1):
    print()
    print("train_bird6 <mname> rgb|gray 64|32 <minibatch> <epochs> <outdir>")
    print()
    print("  <mname>     - model: lenet5|vgg8|resnet18")
    print("  rgb|gray    - RGB or grayscale")
    print("  64|32       - 64x64 or 32x32 images")
    print("  <minibatch> - minibatch size (e.g. 64)")
    print("  <epochs>    - number of training epochs (e.g. 16)")
    print("  <outdir>    - output directory name")
    print()
    exit(0)

mname = sys.argv[1].lower()
itype = sys.argv[2].lower()
isize = int(sys.argv[3])
batch_size = int(sys.argv[4])
epochs = int(sys.argv[5])
outdir = sys.argv[6]

#  Load the requested dataset
if (isize == 64):
    if (itype == "rgb"):
        x_train = np.load("../data/bird6_64_xtrain.npy")
        x_test  = np.load("../data/bird6_64_xtest.npy")
        input_shape = (64,64,3)
    else:
        x_train = np.load("../data/bird6_gray_64_xtrain.npy")
        x_test  = np.load("../data/bird6_gray_64_xtest.npy")
        input_shape = (64,64,1)
else:
    if (itype == "rgb"):
        x_train = np.load("../data/bird6_32_xtrain.npy")
        x_test  = np.load("../data/bird6_32_xtest.npy")
        input_shape = (32,32,3)
    else:
        x_train = np.load("../data/bird6_gray_32_xtrain.npy")
        x_test  = np.load("../data/bird6_gray_32_xtest.npy")
        input_shape = (32,32,1)

ytrain = np.load("../data/bird6_ytrain.npy")
ytest  = np.load("../data/bird6_ytest.npy")
num_classes = 6

#  Scale [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#  Convert labels to one-hot vectors
y_train = keras.utils.to_categorical(ytrain, num_classes)
y_test = keras.utils.to_categorical(ytest, num_classes)

#  Keep some test data for validation
n = int(0.15*len(y_test))
x_test, x_val = x_test[n:], x_test[:n]
y_test, ytest, y_val = y_test[n:], ytest[n:], y_test[:n]

#  Build the model
if (mname == "lenet5"):
    model = LeNet5(input_shape, num_classes)
elif (mname == "vgg8"):
    model = VGG8(input_shape, num_classes)
else:
    model = ResNet18(input_shape, num_classes)
#model.summary()

#  Compile and train
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_val, y_val))

#  Test
pred = model.predict(x_test, verbose=0)
plabel = np.argmax(pred, axis=1)
cm, acc = ConfusionMatrix(plabel, ytest, num_classes=num_classes)
mcc = matthews_corrcoef(ytest, plabel)
txt = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc) 
print(cm)
print(txt)
print()

#  Store results
os.system("mkdir %s 2>/dev/null" % outdir)
with open(outdir+"/accuracy_mcc.txt","w") as f:
    f.write(txt+"\n")
np.save(outdir+"/confusion_matrix.npy", cm)
model.save(outdir+"/model.keras")
terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
x = list(range(epochs))
plt.plot(x, terr, linestyle='solid', linewidth=0.5, color='k', label='train')
plt.plot(x, verr, linestyle='solid', color='k', label='validation')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(outdir+"/error_plot.png", dpi=300)
with open(outdir+"/command_line.txt","w") as f:
    f.write(" ".join(sys.argv)+"\n")

