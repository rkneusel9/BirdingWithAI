#
#  file:  background_chips.py
#
#  Extract 100 64x64 random chips from each of the raw
#  background images
#
#  RTK, 15-Dec-2024
#  Last update:  15-Dec-2024
#
################################################################

import numpy as np
from PIL import Image
import os

#  List of raw images
names = ["raw/"+i for i in os.listdir("raw")]

for raw in range(len(names)):
    img = np.array(Image.open(names[raw]).convert("RGB"))
    rows, cols, _ = img.shape
    for count in range(100):
        x = np.random.randint(0,rows-64)
        y = np.random.randint(0,cols-64)
        im = Image.fromarray(img[x:x+64, y:y+64, :])
        im.save("background/chip_%02d_%03d.png" % (raw,count))

