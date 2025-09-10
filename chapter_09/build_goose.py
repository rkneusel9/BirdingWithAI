#
#  file:  build_goose.py
#
#  Create the Canada and Cackling goose dataset
#
#  RTK, 25-Jan-2025
#  Last update: 25-Jan-2025
#
################################################################

import os
import librosa
import librosa.display
import matplotlib.pylab as plt
import numpy as np
from PIL import Image

canada, cackling = [],[]
for k in range(10):
    #  Canada goose
    y,sr = librosa.load("reference/canada/canada_goose%d.wav" % k)
    duration = 2                    #  2 second windows
    groups = int((len(y)/sr) // duration)
    samp_per_group = duration*sr    #  samples per group
    for i in range(groups):
        offset = i*samp_per_group
        ys = y[offset:(offset+samp_per_group)]
        S = librosa.stft(ys)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        plt.figure(figsize=(2.24,2.24))
        librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap='gray_r')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig("/tmp/ttt.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        im = np.array(Image.open("/tmp/ttt.png").convert("RGB"))
        canada.append(im)

    #  Cackling goose
    y,sr = librosa.load("reference/cackling/cackling_goose%d.wav" % k)
    duration = 2                    #  2 second windows
    groups = int((len(y)/sr) // duration)
    samp_per_group = duration*sr    #  samples per group
    for i in range(groups):
        offset = i*samp_per_group
        ys = y[offset:(offset+samp_per_group)]
        S = librosa.stft(ys)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        plt.figure(figsize=(2.24,2.24))
        librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap='gray_r')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig("/tmp/ttt.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        im = np.array(Image.open("/tmp/ttt.png").convert("RGB"))
        cackling.append(im)

canada = np.array(canada)
cackling = np.array(cackling)

np.random.seed(73939133)  # build the same set each run

idx = np.argsort(np.random.random(len(canada)))
canada = canada[idx]
n = int(0.8*len(canada))
canada_xtrn, canada_xtst = canada[:n], canada[n:]

idx = np.argsort(np.random.random(len(cackling)))
cackling = cackling[idx]
n = int(0.8*len(cackling))
cackling_xtrn, cackling_xtst = cackling[:n], cackling[n:]

xtrn = np.vstack((canada_xtrn, cackling_xtrn))
ytrn = np.array([0]*len(canada_xtrn) + [1]*len(cackling_xtrn))
xtst = np.vstack((canada_xtst, cackling_xtst))
ytst = np.array([0]*len(canada_xtst) + [1]*len(cackling_xtst))

idx = np.argsort(np.random.random(len(xtrn)))
xtrn, ytrn = xtrn[idx], ytrn[idx]
idx = np.argsort(np.random.random(len(xtst)))
xtst, ytst = xtst[idx], ytst[idx]

np.save("goose_xtrain.npy", xtrn)
np.save("goose_ytrain.npy", ytrn)
np.save("goose_xtest.npy", xtst)
np.save("goose_ytest.npy", ytst)

