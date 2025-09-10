#
#  file:  vgg8.py
#
#  VGG8 for bird6
#
#  RTK, 14-Aug-2023
#  Last update:  10-Sep-2024
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
import numpy as np

def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


def ConvBlock(_, filters):
    """Build a VGG convolution-relu-maxpooling block w/optional dropout"""
    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)  # BN before activation
    _ = ReLU()(_)
    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    return MaxPooling2D((2,2))(_)


def DenseBlock(_, nodes):
    """Build a Dense-ReLU-Dropout block"""
    _ = Dense(nodes)(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    return Dropout(0.5)(_)


def VGG8(input_shape, num_classes=10):
    """Return a VGG8 model instance"""
    inp = Input(input_shape)
    _ = ConvBlock(inp, 64)
    _ = ConvBlock(_,  128)
    _ = ConvBlock(_,  256)
    _ = Flatten()(_)
    _ = DenseBlock(_, 2048)
    _ = DenseBlock(_, 2048)
    _ = Dense(num_classes)(_)
    outp = Softmax()(_)
    return Model(inputs=inp, outputs=outp)


#
#  Main:
#
if (__name__ == "__main__"):
    #  Command line
    if (len(sys.argv) == 1):
        print()
        print("vgg8 <minibatch> <epochs> <outdir> [<nsamp>]")
        print()
        print("  <minibatch> - minibatch size (e.g. 128)")
        print("  <epochs>    - number of training epochs (e.g. 16)")
        print("  <outdir>    - output file directory (overwritten)")
        print("  <nsamp>     - number of training samples (optional, all if not given)")
        print()
        exit(0)

    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    outdir = sys.argv[3]
    nsamp = 50000 if len(sys.argv) < 5 else int(sys.argv[4])

    #  Other parameters
    num_classes = 10
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)

    #  Load the CIFAR-10 dataset
    x_train = np.load("cifar10_xtrain.npy")[:nsamp]
    ytrain  = np.load("cifar10_ytrain.npy")[:nsamp]
    x_test  = np.load("cifar10_xtest.npy")
    ytest   = np.load("cifar10_ytest.npy")

    #  Scale [0,1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    #  Convert labels to one-hot vectors
    y_train = keras.utils.to_categorical(ytrain, num_classes)
    y_test = keras.utils.to_categorical(ytest, num_classes)

    #  Keep some training samples for "validation"
    N = int(0.1*len(x_train))
    x_val, x_train = x_train[:N], x_train[N:]
    y_val, y_train = y_train[:N], y_train[N:]

    #  Build the VGG8 model
    model = VGG8(input_shape)
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
    cm, acc = ConfusionMatrix(plabel, ytest)
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

