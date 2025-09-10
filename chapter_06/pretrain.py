#
#  file:  pretrain.py
#
#  Pretrain VGG8 and ResNet-18 models on the Birds 25
#  dataset
#
#  RTK, 04-Nov-2024
#  Last update:  04-Nov-2024
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
import numpy as np
import matplotlib.pylab as plt

from resnet18 import ResNet18, ConfusionMatrix
from vgg8 import VGG8


if (len(sys.argv) == 1):
    print()
    print("pretrain <model> <mb> <epochs> <outdir>")
    print()
    print("  <model>  - 'vgg8' or 'resnet18'")
    print("  <mb>     - minibatch size")
    print("  <epochs> - training epochs")
    print("  <outdir> - output directory")
    print()
    exit(0)

mname = sys.argv[1].lower()
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
outdir = sys.argv[4]

#  Load the Birds 25 datasets
x_train = np.load("../data/birds_25_xtrain.npy")
x_test  = np.load("../data/birds_25_xtest.npy")
ytrain = np.load("../data/birds_25_ytrain.npy")
ytest  = np.load("../data/birds_25_ytest.npy")
input_shape = (64,64,3)
num_classes = 25

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
if (mname == "vgg8"):
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

