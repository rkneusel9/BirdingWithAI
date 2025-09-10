#
#  file:  lenet.py
#
#  LeNet-style model for BirdCLEF
#
#  RTK, 24-Jan-2025
#  Last update:  24-Jan-2025
#
################################################################

import sys
import os
import numpy as np
from sklearn.metrics import matthews_corrcoef
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization

def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

def LeNet(input_shape, num_classes=10):
    """Return a LeNet-style model object"""
    inp = Input(input_shape)
    _ = Conv2D(16, (3,3), padding='same')(inp)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(32, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(64, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Flatten()(_)
    _ = Dense(128)(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Dropout(0.5)(_)
    _ = Dense(num_classes)(_)
    outp = Softmax()(_)
    return Model(inputs=inp, outputs=outp)


if (len(sys.argv) == 1):
    print()
    print("lenet <mb> <epochs> <outdir>")
    print()
    print("  <mb>     - minibatch size")
    print("  <epochs> - training epochs")
    print("  <outdir> - output directory")
    print("  <nsamp>  - number of samples (all if not given)")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
outdir = sys.argv[3]
nsamp = 50000 if len(sys.argv) < 5 else int(sys.argv[4])

#  Load the data
num_classes = 180
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

#  Load the dataset
xtrn = np.load("birdclef_xtrain.npy")[:nsamp]
ytrn = np.load("birdclef_ytrain.npy")[:nsamp]
xtst = np.load("birdclef_xtest.npy")
ytst = np.load("birdclef_ytest.npy")

#  Scale [0,1] -- float16 to maximize memory use
xtrn = xtrn.astype('float16') / 255
xtst = xtst.astype('float16') / 255

#  Convert labels to one-hot vectors
ytrain = keras.utils.to_categorical(ytrn, num_classes)
#ytest = keras.utils.to_categorical(ytst, num_classes)

#  Keep some training samples for validation
n = int(0.1*len(xtrn))
xval, xtrn = xtrn[:n], xtrn[n:]
yval, ytrn = ytrain[:n], ytrain[n:]

#  Configure and train
model = LeNet(input_shape, num_classes=num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(xtrn, ytrn,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(xval, yval))

#  Test
pred = model.predict(xtst, verbose=0)
plabel = np.argmax(pred, axis=1)
cm, acc = ConfusionMatrix(plabel, ytst, num_classes=num_classes)
mcc = matthews_corrcoef(ytst, plabel)
txt = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc) 
print(cm)
print(txt)
print()

#  Keep output
os.system("rm -rf %s 2>/dev/null" % outdir)
os.system("mkdir %s" % outdir)
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/predictions.npy", pred)
np.save(outdir+"/test_labels.npy", ytst)
model.save(outdir+"/lenet.keras")
with open(outdir+"/console.txt","w") as f:
    f.write("Test set accuracy: %0.4f, MCC: %0.4f\n" % (acc,mcc))

