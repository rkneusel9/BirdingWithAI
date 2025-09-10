#
#  file:  compare_results.py
#
#  Compare Ch 5 models with Ch 6 models
#
#  RTK, 02-Oct-2024
#  Last update:  21-Oct-2024
#
################################################################

import numpy as np

def Parse(fname):
    t = [i[:-1].split() for i in open(fname)][0]
    mcc = float(t[-1])
    acc = float(t[-3][:-1])
    return acc,mcc

def display(m0,m1):
    for i in range(6):
        for j in range(6):
            print("%3d  " % m0[i,j], end="")
        print("   ", end="")
        for j in range(6):
            print("%3d  " % m1[i,j], end="")
        print() 
    print()


#  ACC, MCC
for mode in ['rgb','gray']:
    for size in [32,64]:
        for model in ['lenet5','vgg8','resnet18']:
            a0,m0 = Parse("../chapter_05/results/bird6_%s_%d_%s/accuracy_mcc.txt" % (mode, size, model))
            a1,m1 = Parse("../chapter_06/results/bird6_%s_%d_f20_%s/accuracy_mcc.txt" % (mode, size, model))
            print("%4s:%d:% 8s: ACC: %0.5f  =>  %0.5f   MCC: %0.5f  =>  %0.5f" % (mode,size,model,a0,a1,m0,m1))
print()

#  Confusion matrices
for mode in ['rgb','gray']:
    for size in [32,64]:
        for model in ['lenet5','vgg8','resnet18']:
            print("%4s:%d:% 8s:" % (mode, size, model))
            c0 = np.load("../chapter_05/results/bird6_%s_%d_%s/confusion_matrix.npy" % (mode, size, model))
            c1 = np.load("../chapter_06/results/bird6_%s_%d_f20_%s/confusion_matrix.npy" % (mode, size, model))
            display(c0,c1)
print()

