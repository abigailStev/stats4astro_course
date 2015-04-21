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

# from tutorial_ch2: 
morley = np.loadtxt("data/morley.txt", skiprows=1)
speed = morley[:, 2]
rspeed = speed.reshape((5,20))


## 4: Simple statistical inference


## 4.1: Computing the one-sample t-statistic explicitly

s = rspeed[:,0]  # first experiment from Morley data set
mean_s = np.mean(s)
mu = 734.5
se_s = np.sqrt(np.var(s, ddof=1) / float(len(s)))
tstat = (mean_s - mu) / se_s
# print "The t-statistic is: " + str(tstat)

## 4.2: Computing the one-sample t-statistic with a single function

tstat2 = scipy.stats.ttest_1samp(s, popmean=734.5)
# print "The t-statistic is: " + str(tstat2[0])

## 4.3: Computing the two-sample t-statistic

s2 = rspeed[:,1]  # other sample
mean_s2 = np.mean(s2)
se_s2 = np.sqrt(np.var(s2, ddof=1) / float(len(s2)))
twosamp_tstat = (mean_s - mean_s2) / np.sqrt(se_s**2. + se_s2**2.)

# print "2-sample t-statistic: " + str(twosamp_tstat)

twosamp_tstat2 = scipy.stats.ttest_rel(s, s2)
# print "Scipy.stats version of 2-sample t-statistic: " + str(twosamp_tstat2[0])
## Why are they different?? 

## 4.4: Linear regression - generating and plotting the data

x = np.array([10.0, 12.2, 14.4, 16.7, 18.9, 21.1, 23.3, 25.6, 27.8, 30.0])
y = np.array([12.6, 17.5, 19.8, 17.0, 19.7, 20.6, 23.9, 28.9, 26.0, 30.6])

# fig, ax = plt.subplots(1,1,figsize=(6,6))
# ax.plot(x,y,"o")
# ax.set_xlabel("x", fontsize=20)
# ax.set_ylabel("y", fontsize=20)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# ax.set_xlim(0.0, 35.0)
# ax.set_ylim(0.0, 35.0)
# plt.show()

## 4.5: Simple linear regression - the long way

xm = np.mean(x)
ym = np.mean(y)

x2m = np.mean(x**2.)
xym = np.mean(y*x)

b = (xym - xm*ym) / (x2m - xm**2.)
a = ym - b*xm

# fig, ax=plt.subplots(1, figsize=(6,6))
# ax.plot(x,y, "o")
# ax.plot(x, a+b*x, lw=2)
# ax.set_xlabel("x", fontsize=20)
# ax.set_ylabel("y", fontsize=20)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# ax.set_xlim(0.0, 35.0)
# ax.set_ylim(0.0, 35.0)
# plt.show()

## 4.6: Simple linear regression - the short way

# Various ways to do this: np.polyfit(), scipy.optimize.curve_fit(), 
# or in seaborn using regplot() or lmplot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

## first attempt: np.polyfit
r = np.polyfit(x, y, 1)
# ax1.plot(x, y, "o")
# ax1.plot(x, r[0]*x+r[1], lw=2)
## second attempt: scipy.optimize.curve_fit
func = lambda x, a, b: x*a+b
r2, pcov = scipy.optimize.curve_fit(func, x, y, p0=(1,1))
# ax2.plot(x, y, "o")
# ax2.plot(x, r2[0]*x+r2[1], lw=2)
# plt.show()

## 4.7: Linear regression of Reynolds' data

reynolds = np.genfromtxt("data/reynolds.txt", dtype=np.float, names=["dP", "v"], skip_header=1, autostrip=True)
## change units
ppm = 9.80665e3
dp = reynolds["dP"] * ppm
v = reynolds["v"]
## save to new file:
np.savetxt("data/fluid.txt", np.transpose([dp, v]))

popt, pcov = scipy.optimize.curve_fit(func, dp, v)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
# 
# ax1.plot(dp, v, "o")
# ax1.plot(dp, popt[0]*dp+popt[1], lw=2)
# ax1.set_xlabel("Pressure gradient (Pa/m)", fontsize=16)
# ax1.set_ylabel("Velocity (m/s)", fontsize=16)
# ax1.set_title("Fit")
# ax1.tick_params(axis='x', labelsize=16)
# ax1.tick_params(axis='y', labelsize=16)
# 
# ax2.plot(dp, v-(popt[0]*dp+popt[1]), "o")
# #ax2.plot(dp, np.ones(len(dp)), lw=2)
# ax2.set_xlabel("Pressure gradient (Pa/m)", fontsize=16)
# ax2.set_ylabel("Velocity residuals (m/s)", fontsize=16)
# ax2.set_title("Residuals")
# ax2.tick_params(axis='x', labelsize=16)
# ax2.tick_params(axis='y', labelsize=16)
# 
# plt.show()

## 4.8: A closer look ar Reynolds' data

dp_red = dp[:8]
v_red = v[:8]

popt_red, pcov_red = scipy.optimize.curve_fit(func, dp_red, v_red)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.plot(dp_red, v_red, "o")
ax1.plot(dp_red, popt_red[0]*dp_red+popt_red[1], lw=2)
ax1.set(xlabel="dP", ylabel="v")
ax1.set_title("Fit")
ax2.plot(dp_red, v_red-(popt_red[0]*dp_red+popt_red[1]), "o")
ax2.set_title("Residuals")
ax2.set(xlabel="dP", ylabel="v resid")

plt.show()

## 4.9: Numerical diagnostics of the Reynolds' data

sst = np.sum((v_red - np.mean(v_red))**2.)
prediction = popt_red[0]*dp_red+popt_red[1]

ssm = np.sum((prediction - np.mean(v_red))**2.)
sse = np.sum((v_red - prediction)**2.)
r2 = ssm/sst
print "(SSM+SSE)/SST: " + str((ssm+sse)/sst)
print "r: " + str(np.sqrt(r2))
print scipy.stats.pearsonr(dp_red, v_red)





