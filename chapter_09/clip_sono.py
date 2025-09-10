#
#  file:  clip_sono.py
#
#  Compare CLIP embeddings of samples to references
#
#  RTK, 22-Jan-2025
#  Last update: 03-Feb-2025
#
################################################################

import numpy as np
import sys

def Label(n):
    return [
        "Black-bellied Whistling Duck", "Blue-gray Gnatcatcher",
        "Carolina Wren", "Limpkin", "Northern Cardinal",
        "Northern Mockingbird", "Northern Parula", 
        "Red-shouldered Hawk", "Sandhill Crane", "White-eyed Vireo"
    ][n]

def Average(ref, lbl):
    """Average over labels"""
    avg = []
    for i in range(10):
        idx = np.where(lbl==i)[0]
        avg.append(ref[idx].mean(axis=0))
    return np.array(avg), np.arange(10).astype("uint8")

def Cosine(a,b):
    """Return the cosine distance between vectors a and b"""
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return 1.0 - np.dot(a,b) / (ma * mb)

if (len(sys.argv) == 1):
    print()
    print("clip_sono <topn> <mode>")
    print()
    print("  <topn> - top n matches")
    print("  <mode> - 'avg', 'all', 'name'")
    print()
    exit(0)

topn = int(sys.argv[1])
mode = sys.argv[2].lower()

samples = np.load("samples_features.npy")
labels  = np.load("samples_labels.npy")
ref = np.load("reference_features.npy")
lbl = np.load("reference_labels.npy")

if (mode == "avg"):
    ref,lbl = Average(ref,lbl)
elif (mode == "name"):
    ref = np.load("name_features.npy")
    lbl = np.load("name_labels.npy")

for i in range(len(samples)):
    scores = []
    for j in range(len(ref)):
        scores.append(Cosine(ref[j], samples[i]))
    order = np.argsort(scores)[:topn]
    print("%s" % Label(labels[i]))
    for k in order:
        print("(%0.6f)  %s" % (scores[k], Label(lbl[k])))
    print()

