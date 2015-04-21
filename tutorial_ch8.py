import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats.mstats import mquantiles as quantiles
from collections import OrderedDict

## try to import pandas:
try:
	import pandas as pd
	pd_flag = True
except ImportError:
	print "Package pandas not found. Cannot do funky data structures!"
	pd_flag = False
## Seaborn doesn't work with Mac OS X 10.10 and matplotlib -- causes matplotlib to break


## 8: Maximum likelihood estimation


## 8.1: A model comparison for the Rutherford-Geiger data

rutherford = np.loadtxt("data/rutherford.dat")
rate = rutherford[:,0]
freq = rutherford[:,1]
nobs = np.sum(freq)  ## total number of intervals recorded
ntot = np.sum(freq * rate)  ## total number of scintillations counted
meanrate = ntot/nobs
print rate
print freq

def poisson_model(x, l, scale):
	p = scipy.stats.poisson(l)
	return p.pmf(x) * scale

y_rate = poisson_model(rate, l=meanrate, scale=nobs)

fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.vlines(rate, 0, freq, color="b", lw=5, alpha=0.5)
ax.plot(rate, y_rate, "o-", lw=2)
plt.title("Rutherford data plotted with a Poisson model")
plt.show()