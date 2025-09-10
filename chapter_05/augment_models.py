#
#  file:  augment_models.py
#
#  Explore the effect of augmentation
#
#  RTK, 18-Oct-2024
#  Last update:  26-Oct-2024
#
################################################################

import os
import numpy as np

from MCC import MCC

f = [5,10,15,20,25,50,75,100,150]

N, acc, mcc = 6,[],[]
for i in range(len(f)):
    a,m = [],[]
    for k in range(N):
        cmd = "python3 train_bird6_augment.py lenet5 rgb 64 64 32 %d /tmp/ttt" % f[i]
        os.system(cmd)
        cm = np.load("/tmp/ttt/confusion_matrix.npy")
        ac = np.diag(cm).sum() / cm.sum()
        mc = MCC(cm)
        a.append(ac)
        m.append(mc)
    acc.append(a)
    mcc.append(m)
acc = np.array(acc)
mcc = np.array(mcc)
np.save("augment_models_results_acc.npy", acc)
np.save("augment_models_results_mcc.npy", mcc)

print()
print("===============================================")
print("ACC:")

for i in range(len(f)):
    print("(%3d)  %0.6f +/- %0.6f  (%0.6f)" % (f[i],acc[i].mean(),acc[i].std(ddof=1)/np.sqrt(N),np.median(acc[i])))
print()
print("MCC:")

for i in range(len(f)):
    print("(%3d)  %0.6f +/- %0.6f  (%0.6f)" % (f[i],mcc[i].mean(),mcc[i].std(ddof=1)/np.sqrt(N),np.median(mcc[i])))
print()

