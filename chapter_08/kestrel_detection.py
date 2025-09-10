#
#  file:  kestrel_detection.py
#
#  Apply the fully convolutional kestrel model
#  to arbitrary images
#
#  RTK, 16-Dec-2024
#  Last update:  16-Dec-2024
#
################################################################

import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

if (len(sys.argv) == 1):
    print()
    print("kestrel_detection <threshold> <fcn> <image> <outdir>")
    print()
    print("  <threshold> - threshold (keep if above)")
    print("  <fcn>       - fully convolutional kestrel model (.keras)")
    print("  <image>     - image to classify")
    print("  <outdir>    - output directory")
    print()
    exit(0)

threshold = float(sys.argv[1])
model = load_model(sys.argv[2])
orig = np.array(Image.open(sys.argv[3]).convert("RGB"))
outdir = sys.argv[4]

#  Resize the input to a multiple of 64 in each dimension
h,w,c = orig.shape
sx,sy = int(64*np.round(h/64)), int(64*np.round(w/64))
img = Image.fromarray(orig).resize((sx,sy), resample=Image.BILINEAR)
image = np.array(img) / 255

#  Predict over the image
h,w,c = image.shape
pred = model.predict(image.reshape((1,h,w,c)), verbose=0).squeeze()

#  Upscale background and kestrel predictions
#  The model downsamples by a factor of 4: 64x64 to 16x16
#  so zoom by a factor of 4x (bilinear), then place in output
#  with a proper offset
t = zoom(pred[:,:,1], (4,4), order=1)
xx,yy = t.shape
xoff = (h - xx) // 2
yoff = (w - yy) // 2
raw = np.zeros((h,w))
raw[xoff:(xoff+xx), yoff:(yoff+yy)] = t

t = zoom(pred[:,:,0], (4,4), order=1)
xx,yy = t.shape
xoff = (h - xx) // 2
yoff = (w - yy) // 2
background = np.zeros((h,w))
background[xoff:(xoff+xx), yoff:(yoff+yy)] = t

#  Apply threshold
idx = np.where(raw < threshold)
kestrel = raw.copy()
kestrel[idx] = 0.0

#  Alpha-blend detection heat map and original image
gray = Image.fromarray((255*image).astype("uint8")).convert("L")
gray = np.array(gray)

detect = np.zeros((h,w,c), dtype="uint8")
for i in range(3):  detect[:,:,i] = gray
hmap = 255*np.ones((h,w), dtype="uint8")
detect[:,:,0] = kestrel*hmap + (1-kestrel)*detect[:,:,0]
nh,nw = orig.shape[1], orig.shape[0]
detect = Image.fromarray(detect).resize((nh,nw), resample=Image.BILINEAR)

#  Dump output files
os.system("rm -rf %s 2>/dev/null; mkdir %s" % (outdir,outdir))
np.save(outdir+"/background.npy", background)
np.save(outdir+"/kestrel_raw.npy", raw)
np.save(outdir+"/kestrel_detect.npy", kestrel)
detect.save(outdir+"/overlay_image.png")
Image.open(sys.argv[3]).convert("RGB").save(outdir+"/original_image.png")

