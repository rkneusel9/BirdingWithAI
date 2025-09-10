#
#  file:  sono.py
#
#  Use librosa to turn a .wav file into a sonogram
#
################################################################

import librosa
import librosa.display
import matplotlib.pylab as plt
import numpy as np
import sys

if (len(sys.argv) == 1):
    print()
    print("sono <wav> <png>")
    print()
    print("  <wav> - input wav file (clipped at 5 seconds)")
    print("  <png> - output sonogram image")
    print()
    exit(0)

wname = sys.argv[1]
sname = sys.argv[2]

#  Load up to 5 seconds
y,sr = librosa.load(wname, duration=5.0)

#  Generate the sonogram
S = librosa.stft(y)  # short-time Fourier transform
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)  # convert to decibels

#  Plot the sonogram (reverse grayscale)
plt.figure(figsize=(12,9))
librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap='gray_r')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(sname, bbox_inches='tight', pad_inches=0)
plt.close()

