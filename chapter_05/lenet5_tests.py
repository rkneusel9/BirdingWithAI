#
#  file:  lenet5_tests.py
#
#  Unaugmented bird6 with LeNet-5 varying minibatch size
#  and epochs to hold the number of gradient descent steps fixed
#
#  RTK, 18-Oct-2024
#  Last update:  18-Oct-2024
#
################################################################

import os
import numpy as np

m = [1,2,4,8,16,32,64,128]
e = [2,4,8,16,32,64,128,256]

# 32x32
N, acc = 12,[]
for i in range(len(e)):
    a = []
    for k in range(N):
        cmd = "python3 train_bird6.py lenet5 gray 32 %d %d /tmp/ttt" % (m[i],e[i])
        os.system(cmd)
        cm = np.load("/tmp/ttt/confusion_matrix.npy")
        ac = np.diag(cm).sum() / cm.sum()
        a.append(ac)
    acc.append(a)
a32 = np.array(acc)
np.save("lenet5_32_tests.npy", a32)

# 64x64
N, acc = 12,[]
for i in range(len(e)):
    a = []
    for k in range(N):
        cmd = "python3 train_bird6.py lenet5 gray 64 %d %d /tmp/ttt" % (m[i],e[i])
        os.system(cmd)
        cm = np.load("/tmp/ttt/confusion_matrix.npy")
        ac = np.diag(cm).sum() / cm.sum()
        a.append(ac)
    acc.append(a)
a64 = np.array(acc)
np.save("lenet5_64_tests.npy", a64)

print()
print("===============================================")
print()

print("32x32:")
for i in range(len(e)):
    print("(%2d,%2d)  %0.6f +/- %0.6f  (%0.6f)" % (m[i],e[i],a32[i].mean(),a32[i].std(ddof=1)/np.sqrt(N),np.median(a32[i])))
print()

print("64x64:")
for i in range(len(e)):
    print("(%2d,%2d)  %0.6f +/- %0.6f  (%0.6f)" % (m[i],e[i],a64[i].mean(),a64[i].std(ddof=1)/np.sqrt(N),np.median(a64[i])))
print()

