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


## 7: Monte Carlo methods


## 7.1: Pseudo-random number generation

## Default random number generation -- uses dev/urandom if it exists
## otherwise uses the system clock
print np.random.uniform(0, 1, size=10)  ## numpy uniform
print scipy.stats.norm.rvs(0, 1, size=10)  ## scipy normal

## Now set the seed (must be an integer or array of integers)
np.random.seed(101)
print np.random.uniform(0, 1, size=10)
## Now call the function again -- the seed changes every time the RNG is called,
## so the sequence is different
print np.random.uniform(0, 1, size=10)

## So reset the seed again:
np.random.seed(101)
print np.random.uniform(0, 1, size=10)  ## since we set the seed to the same as above, we get the same sequence as above
print np.random.uniform(0, 1, size=10) ## back to the same as above

## And finally reset to the default seed
np.random.seed(None)
print np.random.uniform(0, 1, size=10)


## 7.2: The shape of pseudo-random numbers

## check the distribution with a histogram
unif = np.random.uniform(0, 1, size=1000000)
fig, ax1 = plt.subplots(1, 1, figsize=(8,8))
countsunif, edges = np.histogram(unif, bins=100, range=[0,1], density=True)
ax1.plot(edges[:-1], countsunif, lw=3, color="red", linestyle="steps-mid")
ax1.set_title("Distribution of pseudo-random numbers")
plt.show()


## 7.3: Estimating pi by 'hit and miss' Monte Carlo

npoints = 10000
x = np.random.uniform(0, 1, size=npoints)
y = np.random.uniform(0, 1, size=npoints)
rsq = pow(x,2) + pow(y,2)
naccept = sum (rsq <= 1.0)
area = 4.0 * naccept / npoints
print area

nmax = 100000
area_est = np.empty(nmax)
nsamp = np.empty(nmax)
naccept = 0
i = 0
while (i < nmax):
	x = np.random.uniform(0, 1)
	y = np.random.uniform(0, 1)
	rsq = pow(x,2) + pow(y,2)
	if (rsq <= 1.0): naccept = naccept + 1
	area_est[i] = 4.0 * naccept / (i + 1)
	i += 1
	nsamp[i-1] = i

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(nsamp, area_est, lw=2, color="red")
ax.axis([10, nmax, 3.05, 3.25])
plt.axhline(y = 3.14159265359)
plt.title(r"Converging on $\pi$ using Monte Carlo")
plt.show()


## 7.4: The birthday 'paradox'

days = np.arange(365)
nmatch = 0
npeople = 23
nsim = 1000
i = 0
while i < nsim:
	room = np.random.choice(days, size=npeople, replace=True)
	match = len(room) != len(set(room))  ## The set command removes duplicates!
	if match == True:
		nmatch = nmatch + 1
	i += 1
print nmatch  ## So for 1000 simulations you'll have ~500 cases of a match!

