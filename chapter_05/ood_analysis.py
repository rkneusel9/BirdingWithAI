#
#  file:  ood_analysis.py
#
#  Compare OOD known labels to assigned bird6 classes
#
#  RTK, 23-Oct-2024
#  Last update:  23-Oct-2024
#
#  bird6:
#   (0) American Kestrel
#   (1) American White Pelican
#   (2) Belted Kingfisher
#   (3) Great Blue Heron
#   (4) Red-tailed Hawk
#   (5) Snowy Egret
#
#  ood:
#   (0) Cooper's Hawk
#   (1) Great Egret
#   (2) Swainson's Hawk
#   (3) Black-crowned Night Heron
#   (4) Say's Phoebe
#   (5) Dragonfly
#   (6) Flower
#
################################################################

import os
import sys
import numpy as np

if (len(sys.argv) == 1):
    print()
    print("ood_analysis <assigned> <true>")
    print()
    print("  <assigned> - bird6 assigned classes")
    print("  <true>     - actual OOD labels")
    print()
    exit(0)

labels = np.load(sys.argv[1])
known = np.load(sys.argv[2])
if (len(sys.argv) == 4):
    keep = np.load(sys.argv[3])
else:
    keep = range(len(labels))

known = known[keep]

bird6 = [
    "Kestrel", "Pelican", "Kingfisher", "Blue Heron", "Red-tail", "Snowy Egret"
]

ood = [
    "Cooper", "Great Egret", "Swainson", "Night Heron", "Phoebe", "Dragonfly", "Flower"
]

cm = np.zeros((7,6), dtype="uint8")
for i in range(len(known)):
    cm[known[i],labels[i]] += 1

print(" "*12, end="")
for i in range(6):
    print("%12s" % bird6[i], end="")
print()

for i in range(7):
    print("%14s:" % ood[i], end="")
    for j in range(6):
        print("%9d  " % cm[i,j], end="")
    print()


