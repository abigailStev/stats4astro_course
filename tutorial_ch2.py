import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

## try to import pandas:
try:
	import pandas as pd
	pd_flag = True
except ImportError:
	print "Package pandas not found. Cannot do funky data structures!"
	pd_flag = False

## Seaborn doesn't work with Mac OS X 10.10 and matplotlib -- causes matplotlib to break

morley = np.loadtxt("data/morley.txt", skiprows=1)
# print morley.shape

## 2: Statistical summaries of data: univariate and categorical data

run = morley[:, 1]
speed = morley[:, 2]
experiment = morley[:, 3]

## 2.1: Histograms

## make histogram
counts, edges = np.histogram(speed, bins=10, range=[np.min(speed), np.max(speed)], density=False)

## plot resulting figure
font_prop = font_manager.FontProperties(size=18)

# fig, ax1 = plt.subplots(1,1,figsize=(6,6))
# ax1.plot(edges[:-1], counts, lw=3, color="black", linestyle="steps-mid")

fig, ax1 = plt.subplots(1,1,figsize=(12,6))
bins,edges,patches = ax1.hist(speed, bins=10, range=[np.min(speed), np.max(speed)], normed=False)

ax1.set_xlabel("Speed - 299000 [km/s]", fontproperties=font_prop)
ax1.set_ylabel("Counts per bin", fontproperties=font_prop)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
plt.show()

## 2.2: Bar Charts

rutherford = np.loadtxt("data/rutherford.dat")
rate = rutherford[:, 0]
freq = rutherford[:, 1]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6), sharey=True)
ax1.bar(rate, freq, width=0.8)
ax1.set_xlabel("Rate [counts/interval]")
ax1.set_ylabel("Frequency")
ax2.vlines(rate, 0, freq, color="b", lw=5, alpha=0.5)
ax2.set_xlabel("Rate [counts/interval]")
plt.show()

## 2.3: Mean, median and mode in Python

morley_mn = np.mean(speed) + 299000  ## mean speed with offset added back in
morley_md = np.median(speed) + 299000  ## median speed with offset added back in
# print morley_mn, morley_md

## 2.4: Variance and standard deviation

morley_std = np.std(speed, ddof=1)
morley_var = np.var(speed, ddof=1)
# print morley_std, morley_var

## 2.5: Calculating with subarrays

speed2 = np.array([sp for sp,ex in zip(speed, experiment) if ex == 2.])
v2 = np.var(speed2)
v_all = []
for s in set(experiment):
	v = np.var([sp for sp, ex in zip(speed, experiment) if ex == s])
# 	print "The variance for experiment %i is %.3f" % (s, v)
	v_all.append(v)

mylist = [ 2, "blue", 10.0, [1,2,3]]
for m in mylist:
	pass
# 	print m
	
## 2.6: Tukey's five-number summary

from scipy.stats.mstats import mquantiles as quantiles
from collections import OrderedDict
# print np.median(speed), quantiles(speed, 0.5)


def five_summary(data):
	minimum = np.min(data)
	maximum = np.max(data)
	quants = quantiles(data, [0.25, 0.5, 0.75])
	
	## make ordered dictionary and store values
	d = OrderedDict()
	d["minimum"] = minimum
	d["25% quantile"] = quants[0]
	d["median"] = quants[1]
	d["75% quantile"] = quants[2]
	d["maximum"] = maximum
	
	return d
	
## make a sample
sample = np.random.normal(2,2,size=100)
## call summary function and print
summary = five_summary(sample)
for key in summary.keys():
	pass
# 	print key +": " + str(summary[key])

# print ""

speed_sum = five_summary(speed)
for key in summary.keys():
	pass
# 	print key +": " + str(speed_sum[key])
	
s = pd.Series(sample)
# print s.describe()

speed_series = pd.Series(speed)
# print speed_series.describe()

## 2.7: Standard errors in Python

mn = np.mean(speed)
vr = np.var(speed, ddof=1)
se_byhand = np.sqrt(vr/float(len(speed)))
se = scipy.stats.sem(speed)

# print "SE, by hand: ", se_byhand
# print "Se, scipy.stats: ", se

## 2.8: Standard errors by group

rspeed = speed.reshape((5,20))
rspeed.shape
# print rspeed.shape
mean_all = np.mean(rspeed, axis=1)
var_all = np.mean(rspeed, axis=1)
se_all = scipy.stats.sem(rspeed, axis=1)
se_all_byhand = np.sqrt(np.var(rspeed, axis=1) / rspeed.shape[1])
# print mean_all
# print var_all
# print se_all
# print se_all_byhand

speed_data = {"data":speed, "mean":mean_all, "var":var_all, "se":se_all}

## 2.9: Plotting error bars

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.errorbar(np.arange(len(mean_all))+1, mean_all, yerr=se_all, marker="o", linestyle="")
ax.set_xlim(0.5, 5.5)
ax.set_xlabel("Experiment")
ax.set_ylabel("Speed - 299000 (km/s)")
# plt.show()

## 2.10: Box plots and violin plots

# this uses seaborn, which doesn't work on mac os x 10.10...
# we have to take the transpose of the rspeed matrix to use these features
# trspeed = np.transpose(rspeed)
# fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,6))
# ax1.boxplot(trspeed)
# sns.boxplot(trspeed, names=["Exp 1", "Exp 2", "Exp 3", "Exp 4", "Exp 5"],
# whis=np.inf, color="PuBuGn_d", ax=ax2)
# sns.violinplot(trspeed, inner="stick", color="Purples", bw=1, ax=ax3)
# ax1.set_ylabel("Speed - 299000 (km/s)", fontsize=20)
# ax1.set_xlabel("Experiment")
# ax3.set_xlabel("Experiment")
# plt.show()

## 2.11: Dot charts

d = [0.382, 0.949, 1.00, 0.532, 11.209, 9.449, 4.007, 3.883]
names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
x = np.arange(len(d)) + 0.1

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.plot(d, x,'o')
ax1.hlines(x, [0], d, linestyles='dotted', lw=2)
ax1.set(xlim=(0,12), yticks=x, yticklabels=names);
ax1.set_xlabel("Diameter [Earth diameters]")
ax1.set_title("Distance from Sun")
## sorted by size
dsort = zip(d,names)
dsort.sort()
d_sort, n_sort = zip(*dsort)
ax2.plot(d_sort, x, 'o')
ax2.hlines(x, [0], d_sort, linestyles='dotted', lw=2)
ax2.set(xlim=(0,12), yticks=x, yticklabels=n_sort)
ax2.set_xlabel("Diameter [Earth diameters]")
ax2.set_title("Sorted by size")
plt.show()

plt.close()