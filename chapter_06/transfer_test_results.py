#
#  file:  transfer_test_results.py
#
#  Interpret transfer_test_results.txt
#
#  RTK, 05-Nov-2024
#  Last update:  13-Nov-2024
#
################################################################

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def Parse(s,mode):
    if (mode=='acc'):
        return float(s.split()[3][:-1])
    else:
        return float(s.split()[-1])

lines = [i[:-1] for i in open("transfer_test_results.txt") if i.find("Test set") != -1]

res = {}
for k in ['MLP:VGG8', 'MLP:RESNET18']:
    ac = np.array([Parse(i,'acc') for i in lines[:(3*6)]])
    ac = ac.reshape((6,3))
    mc = np.array([Parse(i,'mcc') for i in lines[:(3*6)]])
    mc = mc.reshape((6,3))
    lines = lines[(3*6):]
    res[k] = (ac,mc)

print("               ACC                   MCC")
for k in ['MLP:VGG8', 'MLP:RESNET18']:
    print("%s:" % k)
    ac,mc = res[k]
    print("   embedding: %0.5f +/- %0.5f    %0.5f +/- %0.5f" % (ac[:,0].mean(), ac[:,0].std(ddof=1)/np.sqrt(6), mc[:,0].mean(), mc[:,0].std(ddof=1)/np.sqrt(6)))
    print("   random   : %0.5f +/- %0.5f    %0.5f +/- %0.5f" % (ac[:,1].mean(), ac[:,1].std(ddof=1)/np.sqrt(6), mc[:,1].mean(), mc[:,1].std(ddof=1)/np.sqrt(6)))
    print("   unraveled: %0.5f +/- %0.5f    %0.5f +/- %0.5f" % (ac[:,2].mean(), ac[:,2].std(ddof=1)/np.sqrt(6), mc[:,2].mean(), mc[:,2].std(ddof=1)/np.sqrt(6)))
    print()

for k in ['MLP:VGG8', 'MLP:RESNET18']:
    v = res[k][0]
    e,r,u = v[:,0], v[:,1], v[:,2]
    t,p = ttest_ind(e,r); _,m = mannwhitneyu(e,r)
    print("%12s: embeddings vs random   : (t=%10.7f, p=%0.8f, u=%0.8f)" % (k,t,p,m))
    t,p = ttest_ind(e,u); _,m = mannwhitneyu(e,u)
    print("%12s: embeddings vs unraveled: (t=%10.7f, p=%0.8f, u=%0.8f)" % (" "*12,t,p,m))
print()

a,b = res['MLP:RESNET18'][0][:,0], res['MLP:VGG8'][0][:,0]
t,p = ttest_ind(a,b); _,m = mannwhitneyu(a,b)
print("MLP ResNet-18 vs VGG8: (t=%10.7f, p=%0.8f, u=%0.8f)" % (t,p,m))
print()

