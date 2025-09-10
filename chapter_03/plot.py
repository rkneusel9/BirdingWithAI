#  A Matplotlib example

import numpy as np
import matplotlib.pylab as plt

x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(x)

plt.plot(x,y, linewidth=0.7, color='k')
plt.plot(x[::50], y[::50], marker='o', 
         linestyle='none', fillstyle='none',
         color='k')

plt.xlabel("$x$")
plt.ylabel("$\\sin(x)$")
plt.title("A sample plot")
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)

plt.savefig("plot.png", dpi=300)
#plt.savefig("plot.eps", dpi=300)
plt.show()

