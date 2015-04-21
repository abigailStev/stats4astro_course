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


## 5: Probability Theory


## 5.1: Random sampling in Python

s = np.arange(20) ## make 20 numbers from 0 to 19
# print np.random.choice(s, size=5, replace=False)
# print np.random.choice(s, size=100, replace=True)


## 5.2: Frequency distribution of a random sample in Python

nsize = 100
x = np.random.choice(s, size=nsize, replace=True)

## pure np histograms, plotting separate
h, bins = np.histogram(x, bins=20, range=[0,20])
# print h
# print bins

## pure np, with bin edges defined
h2, bins2 = np.histogram(x, bins=np.arange(0,21,1))

## plotting for np implementation
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6), sharey=False)
# ax1.bar(bins2[:-1], h2, width=0.8)

## second implementation, plotting while making histogram
# h, bins, patches = ax2.hist(x, bins=20)
# plt.show()

## Or instead of the previous two lines, defining bin edges directly works too!
# h, bins, patches = ax2.hist(x, bins=np.arange(0,21,1))
# plt.show()


## 5.3: Playing cards in Python

## pure np/scipy implementation
face = np.arange(1,14,1)

##define the column names
suit = ["heart", "diamond", "spade", "club"]
## define the type of variable in each column
types = ["float64" for i in range(len(suit))]

## define a structured numpy array
cards = np.array(face, dtype={'names':suit, 'formats':types})

## this is how you call the entire suit "heart" or the second card in the suit "diamonds"
print cards["heart"]
print cards["diamond"][1]

## now with a panda dataframe
d = {'heart' : pd.Series(np.arange(1,14,1), index=range(13)),
	 'diamond' : pd.Series(np.arange(1,14,1), index=range(13)),
	 'spade' : pd.Series(np.arange(1,14,1), index=range(13)),
	 'club' : pd.Series(np.arange(1,14,1), index=range(13)), }
cards_df = pd.DataFrame(d)
colour = cards_df
print colour