import sys
import numpy as np

#  If a seed given, use it, otherwise use whatever
if (len(sys.argv) > 1):
    np.random.seed(int(sys.argv[1]))

#  Accumulate images (x) and associated labels (y)
x,y = [],[]
for batch in range(1,6):
    d = np.fromfile('cifar-10-batches-bin/data_batch_%d.bin' % batch, dtype="uint8")
    for k in range(10000):
        off = k*3073
        label = d[off]
        off += 1
        im = np.zeros((32,32,3), dtype="uint8")
        im[:,:,0] = d[(off+   0):(off+   0+1024)].reshape((32,32))
        im[:,:,1] = d[(off+1024):(off+1024+1024)].reshape((32,32))
        im[:,:,2] = d[(off+2048):(off+1024+2048)].reshape((32,32))
        x.append(im)
        y.append(label)

#  Turn the Python lists into NumPy arrays:
#  x is (50000,32,32,3) and y is (50000,)
x,y = np.array(x), np.array(y)

#  Scramble the order of the images (and labels)
idx = np.argsort(np.random.random(len(x)))
x,y = x[idx], y[idx]

#  Store the training set
np.save("cifar10_xtrain.npy", x)
np.save("cifar10_ytrain.npy", y)

#  Repeat for the test set of 10,000 images
x,y = [],[]
d = np.fromfile('cifar-10-batches-bin/test_batch.bin', dtype="uint8")
for k in range(10000):
    off = k*3073
    label = d[off]
    off += 1
    im = np.zeros((32,32,3), dtype="uint8")
    im[:,:,0] = d[(off+   0):(off+   0+1024)].reshape((32,32))
    im[:,:,1] = d[(off+1024):(off+1024+1024)].reshape((32,32))
    im[:,:,2] = d[(off+2048):(off+1024+2048)].reshape((32,32))
    x.append(im)
    y.append(label)

x,y = np.array(x), np.array(y)
idx = np.argsort(np.random.random(len(x)))
x,y = x[idx], y[idx]

np.save("cifar10_xtest.npy", x)
np.save("cifar10_ytest.npy", y)

