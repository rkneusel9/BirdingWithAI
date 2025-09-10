#
#  file:  metrics.py
#
#  Metrics with birds vs airplanes example
#
#  RTK, 22-Sep-2024
#  Last update:  22-Sep-2024
#
################################################################

import numpy as np

#
#  Metrics:
#
def ConfusionMatrix(pred, y, num_classes=10):
    """Return a confusion matrix"""
    cm = np.zeros((num_classes,num_classes), dtype="uint32")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    return cm

def Metrics(cm): 
    """Confusion matrix metrics"""
    m = {}
    tn,fp,fn,tp = cm.ravel().astype('float64')
    m['TN'],m['FP'],m['FN'],m['TP'] = cm.ravel()
    m['ACC'] = np.diag(cm).sum() / cm.sum()
    m["TPR"] = tp / (tp + fn)
    m["TNR"] = tn / (tn + fp)
    m["PPV"] = tp / (tp + fp)
    m["NPV"] = tn / (tn + fn)
    m["FPR"] = fp / (fp + tn)
    m["FNR"] = fn / (fn + tp)
    m["F1"] = 2.0*m["PPV"]*m["TPR"] / (m["PPV"] + m["TPR"])
    m["MCC"] = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    m["informedness"] = m["TPR"] + m["TNR"] - 1.0
    m["markedness"] = m["PPV"] + m["NPV"] - 1.0 
    n = tp+tn+fp+fn
    po = (tp+tn)/n
    pe = (tp+fn)*(tp+fp)/n**2 + (tn+fp)*(tn+fn)/n**2
    m["kappa"] = (po - pe) / (1.0 - pe)
    return m


#
#  Main:
#
if (__name__ == "__main__"):
    import sys
    import os
    import tensorflow.keras as keras
    from lenet5 import LeNet5

    if (len(sys.argv) == 1):
        print()
        print("metrics <minibatch> <epochs>")
        print()
        print("  <minibatch> - minibatch size (e.g. 64)")
        print("  <epochs>    - number of training epochs (e.g. 16)")
        print()
        exit(0)

    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])

    #  Other parameters
    num_classes = 2
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)

    #  Load the CIFAR-10 dataset and keep
    #  only airplanes (0) and birds (2 --> 1)
    x_train = np.load("cifar10_xtrain.npy")
    ytrain  = np.load("cifar10_ytrain.npy")
    x_test  = np.load("cifar10_xtest.npy")
    ytest   = np.load("cifar10_ytest.npy")

    #  Rebuild the training set coding birds as class 1
    i0,i2 = np.where(ytrain==0)[0], np.where(ytrain==2)[0]
    x = np.vstack((x_train[i0], x_train[i2]))
    y = np.array([0]*len(i0) + [1]*len(i2))
    idx = np.argsort(np.random.random(len(y)))
    x_train, ytrain = x[idx], y[idx]

    #  Repeat for the test set
    i0,i2 = np.where(ytest==0)[0], np.where(ytest==2)[0]
    x = np.vstack((x_test[i0], x_test[i2]))
    y = np.array([0]*len(i0) + [1]*len(i2))
    idx = np.argsort(np.random.random(len(y)))
    x_test, ytest = x[idx], y[idx]

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

    #  Build the model
    model = LeNet5(input_shape, num_classes=num_classes)

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

    #  Calculate and display metrics
    cm = ConfusionMatrix(plabel, ytest, num_classes=2)
    m = Metrics(cm)
    print()
    print("TN:%4d  FP:%4d" % (m['TN'],m['FP']))
    print("FN:%4d  TP:%4d" % (m['FN'],m['TP']))
    print()
    print("ACC: %0.5f" % m['ACC'], '  (accuracy)')
    print("MCC: %0.5f" % m['MCC'], '  (Matthews correlation coefficient)')
    print()
    print("TPR: %0.5f" % m['TPR'], '  (true positive rate, sensitivity, recall, hit rate)')
    print("TNR: %0.5f" % m['TNR'], '  (true negative rate, specificity)')
    print("FPR: %0.5f" % m['FPR'], '  (false positive rate)')
    print("FNR: %0.5f" % m['FNR'], '  (false negative rate)')
    print()
    print("PPV: %0.5f" % m['PPV'], '  (positive predictive value, precision)')
    print("NPV: %0.5f" % m['NPV'], '  (negative predictive value)')
    print()
    print("F1           : %0.5f" % m['F1'])
    print("informedness : %0.5f" % m['informedness'])
    print("markedness   : %0.5f" % m['markedness'])
    print("kappa        : %0.5f" % m['kappa'])
    print()

