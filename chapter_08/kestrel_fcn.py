#
#  file:  kestrel_fcn.py
#
#  Create fully convolutional version of the base kestrel
#  model
#
#  RTK, 15-Dec-2024
#  Last update:  15-Dec-2024
#
################################################################

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # quiet TF
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization

def LeNet(input_shape, num_classes=10):
    """Return a fully convolutional LeNet-style model object"""
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
    _ = Conv2D(128, (16,16))(_)         # was Dense(128)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Dropout(0.5)(_)
    _ = Conv2D(num_classes, (1,1))(_)   # Dense(num_classes)
    outp = Softmax()(_)
    return Model(inputs=inp, outputs=outp)

if (len(sys.argv) == 1):
    print()
    print("kestrel_fcn <base> <fcn>")
    print()
    print("  <base> - trained base model (.keras)")
    print("  <fcn>  - output fully convolutional model (.keras)")
    print()
    exit(0)

#  Build the fully convolutional model
model = LeNet((None,None,3), 2)

#  Load base model
base = load_model(sys.argv[1])

#  Copy the unaltered weights and biases
for i in [1,2,5,6,9,10]:
    w = base.layers[i].get_weights()
    model.layers[i].set_weights(w)

#  Copy final batch norm layer
w = base.layers[14].get_weights()
model.layers[13].set_weights(w)

#  Copy dense weights reshaping as needed
w = base.layers[13].get_weights()
model.layers[12].set_weights([w[0].reshape([16,16,64,128]), w[1]])
w = base.layers[17].get_weights()
model.layers[16].set_weights([w[0].reshape([1,1,128,2]), w[1]])

#  Output the fully convolutional model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam())
model.save(sys.argv[2])

