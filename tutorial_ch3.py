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

## 3: Statistical summaries of data: bivariate data

## 3.1: Reformatting and cleaning the Hipparcos data

hipparcos = np.genfromtxt("data/hipparcos.txt", dtype=np.float, skip_header=53, skip_footer=2, autostrip=True)
h = pd.DataFrame(hipparcos[:,1:], index=hipparcos[:,0], columns=["Rah", "Ram", "Ras", "DECd", "DECm", "DECs", "Vmag", "Plx", "ePlx", "BV", "eBV"])
# print len(h)
hnew = h[:].dropna(how="any") # get rid of NaNs
# print len(hnew)

# get rid of data with parallax error > 5%
hclean = hnew[hnew.ePlx/np.abs(hnew.Plx) < 0.05]
hclean = hclean[["Vmag", "Plx", "ePlx", "BV", "eBV"]]
hclean["dist"] = 1.e3/hclean["Plx"]  # Convert parallax to distance in pc
# Convert absolute magnitude using distance modulus
hclean["Vabs"] = hclean.Vmag = 5. * (np.log10(hclean.dist) - 1.)
hclean.to_csv("hip_clean.csv")

## 3.2: Scatter plots in Python

## pandas version of reading in the data
hip_pd = pd.read_csv("hip_clean.csv", index_col=0)

fig, ax1 = plt.subplots(1,1,figsize=(6,6))
ax1.scatter(hip_pd.BV, hip_pd.Vabs, c="red")
ax1.set_xlim(-0.3, 2.0)
ax1.set_ylim(16,-3)
ax1.set_xlabel("B-V (mag)")
ax1.set_ylabel("V absolute magnitude (mag)")
plt.show()

## 3.3: Adding 'rugs' to plots
x = np.random.chisquare(2, size=50)
y = x + np.random.normal(size=50)

fig, ax1 = plt.subplots(1,1,figsize=(6,6))
ax1.plot(x, y, "o", color="blue")
ax1.set(xlim=(0.9*np.min(x), 1.1*np.max(x)), ylim=(0.9*np.min(y), 1.1*np.max(y)))
ax1.vlines(x, 0.9*np.min(y), 0.9*np.min(y)+[0.1*(np.max(y)-np.min(y))]*len(x), color='black')
ax1.hlines(y, 0.9*np.min(x), 0.9*np.min(x)+[0.1*(np.max(x)-np.min(x))]*len(y), color='black')
plt.show()

## 3.4: Computing the correlation coefficient
x = np.random.chisquare(2, size=50)
y = x + np.random.normal(size=50)
(cor, pval) = scipy.stats.pearsonr(x, y)
# print cor, pval

## 3.5: A scatter-plot matrix
rand_data = np.random.multivariate_normal([1,2,6,4], np.array([[3,2,1,3], [1,2,1,4], [1,2,3,2], [3,1,3,1]]), size=100)
ndims = rand_data.shape[1]

## As an additional exercise, you could try also calculating the correlation 
## coefficients between each pair of variables. You should be able to find a 
## way to print these coefficients on the corresponding panels in the plot, 
## which adds efficiently to the information shown in the plot.

fig, axes = plt.subplots(4,4,figsize=(10,10))

for i in xrange(ndims): ## x dimension
	for j in xrange(ndims): ## y dimension
		(cor, pval) = scipy.stats.pearsonr(rand_data[:,i], rand_data[:,j])
		if i == j:
			axes[i,j].hist(rand_data[:,i], bins=20)
		else:
			x = rand_data[:,i]
			y = rand_data[:,j]
			(cor, pval) = scipy.stats.pearsonr(x, y)
			axes[i,j].scatter(x, y)
			x_loc = np.min(x) + 0.05 * (np.max(x)-np.min(x))
			y_loc = np.max(y) - 0.01 * (np.max(y)-np.min(y))
			print x_loc, y_loc
			axes[i,j].text(x_loc, y_loc, "r = %.2f" % cor)
		
plt.show()

plt.close()