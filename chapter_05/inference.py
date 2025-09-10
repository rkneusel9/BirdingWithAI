#
#  file:  inference.py
#
#  Run inference on a dataset using a threshold
#
#  RTK, 23-Oct-2024
#  Last update:  23-Oct-2024
#
################################################################

import sys
import os
from io import StringIO
import numpy as np
from tensorflow.keras.models import load_model
from lenet5 import ConfusionMatrix

#  Capture output
text = StringIO()
sysout = sys.stdout

def Print(*args, sep=' ', end='\n'):
    output = sep.join(map(str, args)) + end
    sysout.write(output)
    text.write(output)

if (len(sys.argv) == 1):
    print()
    print("inference <x> <y> <model> <threshold> <outdir>")
    print()
    print("  <x>         - samples to test")
    print("  <y>         - known labels or 'none'")
    print("  <model>     - appropriate trained bird6 model")
    print("  <threshold> - threshold or 0 for none")
    print("  <outdir>    - output directory (overwritten)")
    print()
    exit(0)

xname = sys.argv[1]
yname = sys.argv[2]
mname = sys.argv[3]
threshold = float(sys.argv[4])
outdir = sys.argv[5]

#  Load datasets and model
xtest = np.load(xname) / 255
ytest = np.load(yname) if (yname != "none") else None
model = load_model(mname)

#  Get the model's predictions
pred = model.predict(xtest, verbose=0)
plabel = np.argmax(pred, axis=1)  # no threshold labels

#  Find predictions to label and ignore
ignore = np.where(pred.max(axis=1) < threshold)[0]
keep = np.where(pred.max(axis=1) >= threshold)[0]
ni,nk = len(ignore), len(keep)

#  If known labels given
if (ytest is not None):
    num_classes = ytest.max() + 1  # assumes all classes present
    cm0,acc0 = ConfusionMatrix(plabel, ytest, num_classes=num_classes)
    p,y = pred[keep], ytest[keep]
    cm1,acc1 = ConfusionMatrix(np.argmax(p,axis=1), y, num_classes=num_classes)

#  Report
os.system("rm -rf %s; mkdir %s" % (outdir, outdir))
labeled = np.zeros(len(pred), dtype="uint8")
labeled[keep] = 1
np.save(outdir + "/labeled.npy", labeled)
np.save(outdir + "/ignore_indices.npy", ignore)
np.save(outdir + "/ignore_predictions.npy", pred[ignore])
np.save(outdir + "/keep_indices.npy", keep)
np.save(outdir + "/keep_predictions.npy", pred[keep])
np.save(outdir + "/keep_labels.npy", np.argmax(pred[keep], axis=1))

Print("Assigning %d labels (%0.5f) (threshold=%0.6f)" % (nk, nk/(ni+nk), threshold))

if (ytest is not None):
    Print("No threshold:")
    Print(np.array2string(cm0))
    Print("Overall accuracy = %0.5f\n" % acc0)
    Print("With threshold:")
    Print(np.array2string(cm1))
    Print("Overall accuracy = %0.5f\n" % acc1)
else:
    kl = np.argmax(pred[keep], axis=1)
    klc = np.bincount(kl, minlength=6)
    il = np.argmax(pred[ignore], axis=1)
    ilc = np.bincount(il, minlength=6)
    Print("keep  : %s  (%4d)" % (np.array2string(klc), klc.sum()))
    Print("ignore: %s  (%4d)" % (np.array2string(ilc), ilc.sum()))

with open(outdir + "/console.txt","w") as f:
    f.write(text.getvalue())

