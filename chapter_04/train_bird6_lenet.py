#
#  file:  train_bird6_lenet.py
#
#  Train LeNet-5 RGB 32|64 models many times and histogram
#  the results
#
#  RTK, 03-Oct-2024
#  Last update: 03-Oct-2024
#
################################################################

import os
import numpy as np
import matplotlib.pylab as plt
from lenet5 import LeNet5

N = 300

a32 = []
for i in range(N):
    cmd = "rm -rf /tmp/ttt; python3 train_bird6.py lenet5 rgb 32 16 32 /tmp/ttt"
    os.system(cmd + " >/dev/null 2>/dev/null")
    t = [i[:-1] for i in open("/tmp/ttt/accuracy_mcc.txt")][0]
    a32.append(float(t.split()[3][:-1]))
a32 = np.array(a32)

a64 = []
for i in range(N):
    cmd = "rm -rf /tmp/ttt; python3 train_bird6.py lenet5 rgb 64 16 32 /tmp/ttt"
    os.system(cmd + " >/dev/null 2>/dev/null")
    t = [i[:-1] for i in open("/tmp/ttt/accuracy_mcc.txt")][0]
    a64.append(float(t.split()[3][:-1]))
a64 = np.array(a64)

np.save("results/train_bird6_lenet_a32.npy", a32)
np.save("results/train_bird6_lenet_a64.npy", a64)

h,x = np.histogram(a32, bins=20)
x = 0.5*(x[1:]+x[:-1])
plt.bar(x,h, fill=False, color='k', width=0.9*(x[1]-x[0]))
plt.xlabel("Test set accuracy")
plt.ylabel("Count")
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("lenet5_rgb_32.eps", dpi=300)
plt.savefig("lenet5_rgb_32.png", dpi=300)
plt.close()

h,x = np.histogram(a64, bins=20)
x = 0.5*(x[1:]+x[:-1])
plt.bar(x,h, fill=False, color='k', width=0.9*(x[1]-x[0]))
plt.xlabel("Test set accuracy")
plt.ylabel("Count")
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("lenet5_rgb_64.eps", dpi=300)
plt.savefig("lenet5_rgb_64.png", dpi=300)
plt.close()

