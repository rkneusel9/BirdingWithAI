#
#  file:  build_datasets.py
#
#  Use librosa to turn .wav files into a sonograms to build
#  the reference and samples datasets
#
#  RTK, 22-Jan-2025
#  Last update: 22-Jan-2025
#
################################################################

import os
import torch
import clip
import librosa
import librosa.display
import matplotlib.pylab as plt
import numpy as np
from PIL import Image

def Label(wname):
    """Convert name to label number"""
    if ("black-bellied_whistling_duck" in wname): return 0
    elif ("blue-gray_gnatcatcher" in wname):      return 1
    elif ("carolina_wren" in wname):              return 2
    elif ("limpkin" in wname):                    return 3
    elif ("northern_cardinal" in wname):          return 4
    elif ("northern_mockingbird" in wname):       return 5
    elif ("northern_parula" in wname):            return 6
    elif ("red-shouldered_hawk" in wname):        return 7
    elif ("sandhill_crane" in wname):             return 8
    elif ("white-eyed_vireo" in wname):           return 9
    else: raise ValueError("unknown")

def Sonogram(wname):
    """Convert a .wav file to a sonogram"""
    y,sr = librosa.load(wname, duration=5.0)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(3.36, 3.36))
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap='gray_r')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("/tmp/ttt.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    return np.array(Image.open("/tmp/ttt.png").convert("L"))

def Features(sono, model, preprocess):
    """Generate CLIP embedding"""
    img = Image.fromarray(sono).convert("RGB")
    im = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(im)
    return features.cpu().numpy().squeeze()

#  Configure CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

#  Reference clips
wnames = sorted(["reference/clips/"+i for i in os.listdir("reference/clips")])
sono,labels,features = [],[],[]
for wname in wnames:
    sono.append(Sonogram(wname))
    labels.append(Label(wname))
    features.append(Features(sono[-1], model, preprocess))
    print("%d : %s" % (labels[-1], os.path.basename(wname)), flush=True)
np.save("reference_sonograms.npy", np.array(sono))
np.save("reference_labels.npy", np.array(labels))
np.save("reference_features.npy", np.array(features))

#  Sample clips
print()
wnames = sorted(["samples/wav/"+i for i in os.listdir("samples/wav")])
sono,labels,features = [],[],[]
for wname in wnames:
    sono.append(Sonogram(wname))
    labels.append(Label(wname))
    features.append(Features(sono[-1], model, preprocess))
    print("%d : %s" % (labels[-1], os.path.basename(wname)), flush=True)
np.save("samples_sonograms.npy", np.array(sono))
np.save("samples_labels.npy", np.array(labels))
np.save("samples_features.npy", np.array(features))

#  Name embeddings
names = [
    "Black-bellied Whistling Duck", "Blue-gray Gnatcatcher",
    "Carolina Wren", "Limpkin", "Northern Cardinal",
    "Northern Mockingbird", "Northern Parula", 
    "Red-shouldered Hawk", "Sandhill Crane", "White-eyed Vireo"
]
features = []
for name in names:
    tokens = clip.tokenize([name]).to(device)
    with torch.no_grad():
        f = model.encode_text(tokens).cpu().numpy().squeeze()
    features.append(f)
np.save("name_features.npy", np.array(features))
np.save("name_labels.npy", np.arange(10).astype("uint8"))

