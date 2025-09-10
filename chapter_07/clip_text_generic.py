#
#  file:  clip_text_generic.py
#
#  Use text prompts to locate matching species
#  (zero-shot classification)
#
#  RTK, 29-Nov-2024
#  Last update: 29-Nov-2024
#
################################################################

import sys
import torch
import clip
import numpy as np

def Cosine(a,b):
    """Return the cosine distance between vectors a and b"""
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return 1.0 - np.dot(a,b) / (ma * mb)


if (len(sys.argv) == 1):
    print()
    print("clip_text_generic <top-n> <prompt> [<negate>]")
    print()
    print("  <top-n>  - capture the top-n best matches")
    print("  <prompt> - text describing the bird")
    print("  <negate> - 'negate' to keep the top-n _worst_ matches")
    print()
    exit(0)

topn = int(sys.argv[1])
prompt = sys.argv[2]
negate = (len(sys.argv) == 4)

#  Load the NA birds common name embeddings
num_classes = 404
names= np.load("nabirds_names.npy")
x = np.load("nabirds_name_embeddings.npy")
y = np.arange(num_classes, dtype="uint16")

#  Configure CLIP and get prompt embedding
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

tokens = clip.tokenize([prompt]).to(device)
with torch.no_grad():
    prompt_features = model.encode_text(tokens).cpu().numpy().squeeze()

#  Find the top-n best or worst image embedding matches
#  using cosine distance
scores = []
for i in range(len(x)):
    scores.append(Cosine(x[i], prompt_features))
scores = np.array(scores)

#  Sort everything together to extract scores, labels, and images
#  keeping the top-n smallest cosine distances
if (negate):
    order = np.argsort(scores)[::-1][:topn]
else:
    order = np.argsort(scores)[:topn]
scores = scores[order][:topn]
y = y[order][:topn]

#  Dump the results
s = "(negated)" if negate else ""
print("Prompt: %s %s" % (prompt,s))
for i in range(len(scores)):
    print("(%0.6f)  %s" % (scores[i], names[y[i]]))
print()

