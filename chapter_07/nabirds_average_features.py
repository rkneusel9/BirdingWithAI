#  Create averaged NA birds embeddings to avoid doing it
#  on each run of clip_image_generic.py

import numpy as np

N = np.load("nabirds_name_embeddings.npy")
X = np.load("nabirds_features.npy")
Y = np.load("nabirds_labels.npy")

#  Average the image embeddings
x,y = [],[]
for c in range(404):
    t = X[np.where(Y==c)[0]]
    x.append(t.mean(axis=0))
    y.append(c)
x,y = np.array(x), np.array(y)
np.save("nabirds_averaged_features.npy", x)

#  Average the image embeddings and include the name embedding
x,y = [],[]
for c in range(404):
    t = X[np.where(Y==c)[0]]
    t = np.vstack((t, N[c]))
    x.append(t.mean(axis=0))
    y.append(c)
x,y = np.array(x), np.array(y)
np.save("nabirds_averaged_name_features.npy", x)

