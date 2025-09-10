#
#  file:  birdclef_fine-tune.py
#
#  Fine-tune MobileNetV3 to generate features for transfer learning
#
#  RTK, 07-Feb-2025
#  Last update:  07-Feb-2025
#
################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pickle
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, ReLU, Softmax

def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


if (len(sys.argv) == 1):
    print()
    print("birdclef_fine-tune <mb> <epochs> <outdir>")
    print()
    print("  <mb>     - minibatch size (e.g., 32)")
    print("  <epochs> - fine-tuning epochs (e.g., 9)")
    print("  <outdir> - output directory")
    print()
    exit(0)

mb = int(sys.argv[1])
epochs = int(sys.argv[2])
outdir = sys.argv[3]

#  Load the train and test sonograms
x_train = np.load("birdclef_xtrain.npy")
ytrain  = np.load("birdclef_ytrain.npy")
x_test = np.load("birdclef_xtest.npy")
ytest  = np.load("birdclef_ytest.npy")

#  Load the pretrained model
input_shape = (224, 224, 3)
base = MobileNetV3Large(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')

#  Allow only the higher-level weights to adapt
N = 80  # can be adjusted
for layer in base.layers[:N]: 
    layer.trainable = False
for layer in base.layers[N:]: 
    layer.trainable = True

#  Classification head
num_classes = 180
_ = Dense(256)(base.output)
_ = ReLU()(_)
_ = Dropout(0.5)(_)
_ = Dense(num_classes)(_)
outp = Softmax()(_)

#  Combined model - sparse cross-entropy uses integer labels
model = Model(inputs=base.input, outputs=outp)
model.compile(optimizer=Adam(), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#  Fine-tune with the training set
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

model.fit(x_train, ytrain,
          validation_split=0.2, 
          epochs=epochs, batch_size=mb,
          verbose=1)

#  Predict using the fine-tuned model
prob = model.predict(x_test, verbose=0)
plabel = np.argmax(prob, axis=1)
cm,acc = ConfusionMatrix(plabel, ytest, num_classes=num_classes)
print("Test set accuracy = %0.5f" % acc)

#  Now generate embeddings en masse using the base model
#  (which had its weights updated by fine-tuning)
xtrn = base.predict(x_train, verbose=0)
xtst = base.predict(x_test, verbose=0)

#  Scale the embedding vectors to [0,1]
xtrn = (xtrn - xtrn.min()) / (xtrn.max() - xtrn.min())
xtst = (xtst - xtst.min()) / (xtst.max() - xtst.min())

#  And store on disk
os.system("rm -rf %s 2>/dev/null; mkdir %s" % (outdir,outdir))
np.save(outdir+"/xtrain.npy", xtrn)
np.save(outdir+"/ytrain.npy", ytrain)
np.save(outdir+"/xtest.npy", xtst)
np.save(outdir+"/ytest.npy", ytest)
np.save(outdir+"/confusion_matrix.npy", cm)

