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

## 6: Random Variables


## 6.1: Simulating repeated Bernoulli trials

omega = ["red", "green"]
n = 5
nsims = 1000
xsims = []
for i in range(nsims):
	samp = np.random.choice(omega, size=n, p=[0.6, 0.4], replace=True)
	xsims.append(np.sum([samp == "red"]))  ## sum here turns True and False into 1 and 0, respectively
# print xsims


## 6.2: The binomial distribution emerges

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
## histogram of the samples from the box above
h, bins, patches = ax1.hist(xsims, bins=np.arange(0.0, 50.0, 1.0))
ax1.set_title("1000 random samples")
p = 0.6  ## probability of red sweets
## define binomial distribution of a given sample size n and probability p;
## freezes distribution
b = scipy.stats.binom(n,p)
x = np.arange(0.0, 50., 1.0)  ## x-axis
ax2.plot(x, b.pmf(x), 'bo', ms=8)  ## plot probability mass function for points in x
ax2.vlines(x, 0, b.pmf(x), colors='b', lw=5, alpha=0.5)
ax2.set_title("Probability mass function of a binomial distribution")
plt.show()


## 6.3: Probability distributions

a = np.random.rand(10)  ## 10 numbers from a uniform distribution between 0 and 1
b = np.random.randn(10)  ## 10 numbers from a standard normal distribution with mean=0 and std deviation = 1
sigma = 2
mu = 1
b_scaled = np.random.randn(10) * sigma + mu  ## 10 numbers from a normal distribution with mean=1 and std deviation=2
c = np.random.random_integers(0, 14, size=10)  ## 10 random integers between 0 and 14, inclusive
chi = np.random.chisquare(2, size=10)  ## 10 samples from a chisquared distribution with dof=2
ln = np.random.lognormal(1, 2, size=10)  ## 10 samples from a lognormal distribution with mean=1 and std deviation=2
poisson = np.random.poisson(4, size=10)  ## 10 samples from a poisson distribution with rate parameter lambda=4
norm = np.random.normal(1, 2, size=10)  ## 10 samples from a normal distribution with mean=1 and std deviation=2

## define parameters for normal distribution
mu = 1
sigma = 2
## freeze the distribution for a given mean and standard deviation
n = scipy.stats.norm(mu, sigma)
## now you can use it for all sorts of funky things!
## like compute moments! Computes mean, variance, skew, and kurtosis!
mean, var, skew, kurt = n.stats(moments='mvsk')
## or separately: 
mean = n.mean()
median = n.median()
var = n.var()
std = n.std()
(lower, upper) = n.interval(0.5)  ## interval that contains 0.5 of the distribution

## or you can plot the probability density function
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
x = np.arange(-5.0, 8.0, 0.1)
ax1.plot(x, n.pdf(x), lw=2)
ax1.set_title("Probability distribution function")
## or you can plot the cumulative distribution function:
ax2.plot(x, n.cdf(x), lw=2)
ax2.set_title("Cumulative distribution function")
plt.show()

## or generate random numbers:
samp = n.rvs(size=10)
## Poisson distribution
rpar = 15  ## rate parameter for poisson distribution
## freeze Poisson distribution
p = scipy.stats.poisson(rpar)
x = np.arange(0.0, 30.0, 1.0)

## unlike the normal distribution, the poisson distribution is discrete so it 
## doesn't have a probability density function, it has a probability mass
## function instead! Remember this, or try and call p.pdf(x) and see what happens.
## p.pdf(x)  -> AttributeError: 'poisson_gen' object has no attribute 'pdf'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.plot(x, p.pmf(x), "bo", ms=8)  ## ms is the size of the dot at the end of the line
ax1.vlines(x, 0, p.pmf(x), colors='b', lw=5, alpha=0.5)  ## alpha is transparency
ax1.set_title("Probability mass function")
## other methods work in analogy to the continuous distributions
ax2.plot(x, p.cdf(x))
ax2.set_title("Cumulative distribution function")
plt.show()


## 6.4: The binomial distribution in scipy.stats

x = np.arange(0., 20., 1.)
nsamp = 30  ## sample size
p = 0.1  ## probability for success
b = scipy.stats.binom(nsamp, p)  ## freeze binomial distribution
probmass = b.pmf(x)  ## compute PMF at locations x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.vlines(x, 0, probmass, color='b', lw=5)
ax1.set_title("Probability mass function")
randsamp = b.rvs(size=1000)
bins, edges, patches = ax2.hist(randsamp, bins=np.arange(-0.5, 20.5, 1.0), normed=False)
ax2.set_title("1000 samples from the binomial distribution")
plt.show()


## 6.5: The Poisson distribution

ratepar = 1.5  ## rate parameter for the Poisson distribution
p = scipy.stats.poisson(ratepar)  ## freeze distribution
## compute pmf and cdf for given values
print "Probability of 3 events per day: p = " + str(p.pmf(3))
print "Probability of <=3 events per day: p = " + str(p.cdf(3))
## compute random numbers
ratepar = 3.5  ## new rate parameter
p = scipy.stats.poisson(ratepar)
randsamp=p.rvs(100)  ## generates random numbers from the distribution
fig, ax = plt.subplots(1, 1, figsize=(6,6))
bins, edges, patches = ax.hist(randsamp, bins=np.arange(-0.5, 20.5, 1.0), normed=True)
ax.plot(x, p.pmf(x), lw=3)  ## draws the line on top of the histogram
ax.set_title("Poisson distribution")
plt.show()


## 6.6: The normal distribution

mu = 1  ## mean of distribution
sigma = 2  ## standard deviation of the distribution
nd = scipy.stats.norm(mu, sigma)  ## freeze distribution

randsamp = nd.rvs(5000)
fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
bins, edges, patches = ax1.hist(randsamp, bins=np.arange(-5, 10, 1.0), normed=True)
x = np.arange(-5, 10, .1)
ax1.plot(x, nd.pdf(x), lw=3)
ax1.set_title("Normal distribution")
plt.show()

## 6.7: The chisquared distribution

x = np.arange(0, 20, 0.1)
df = 5  ## dof
ch = scipy.stats.chi2(df)  ## freeze the distribution
randsamp = ch.rvs(5000)  ## generates random numbers from the distribution

fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
bins, edges, patches = ax1.hist(randsamp, bins=np.arange(0,20, 1.0), normed=True)
ax1.plot(x, ch.pdf(x), lw=3)
ax1.set_title("Chisquared distribution")
plt.show()


## 6.8: Adding chisquared distributed data

x = np.arange(0.1, 30, 1.)
samp1 = scipy.stats.chi2.rvs(5, size=50000)  ## generating random numbers from the distribution
samp2 = scipy.stats.chi2.rvs(4, size=50000)  ## generating random numbers from the distribution
z = samp1 + samp2  ## adding the two chisquared-distributed-random numbers together

fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
bins, edges, patches = ax1.hist(z, bins=np.arange(0, 30, 1.0), normed=True)
c3 = scipy.stats.chi2(9)  ## chisquare distribution with 9 (=4+5) dof
ax1.plot(x, c3.pdf(x), lw=3)
ax1.set_title("Adding chisquared distributed data")
plt.show()


## 6.9: Testing the central limit theorem

nvals = 10
nsims = 10000
chisqsims = np.empty(nsims)
unifsims = np.empty(nsims)
mu = 1
sigma = 1 / np.sqrt(nvals)
n = scipy.stats.norm(mu, sigma)

for i in range(nsims):
	chisq = scipy.stats.chi2.rvs(1, size=nvals)  ## generates random numbers from the distribution
	unif = scipy.stats.uniform.rvs(loc=1.0-np.sqrt(3), scale=2.*np.sqrt(3), size=nvals)  ## generates random numbers from the distribution
	chisqsims[i] = np.mean(chisq) ## arithmetic mean of those random numbers
	unifsims[i] = np.mean(unif)  ## arithmetic mean of those random numbers

fig, ax1 = plt.subplots(1, 1, figsize=(8,8))
countschi, edges = np.histogram(chisqsims, bins=30, range=[np.min(chisqsims), np.max(chisqsims)], density=True)
ax1.plot(edges[:-1], countschi, lw=3, color="black", linestyle="steps-mid", label="Chisquared")
countsunif, edges = np.histogram(unifsims, bins=30, range=[np.min(unifsims), np.max(unifsims)], density=True)
ax1.plot(edges[:-1], countsunif, lw=3, color="red", linestyle="steps-mid", label="Uniform")
x = np.arange(-1.0, 3.0, 0.01)
ax1.plot(x, n.pdf(x), lw=2, color="blue", label="Normal pdf")
ax1.set_title("Testing the central limit theorem")
legend = ax1.legend(loc='upper right')
for label in legend.get_texts():
	label.set_fontsize('small')
for label in legend.get_lines():
	label.set_linewidth(2)  # the legend line width
		
plt.show()


## 6.10: The lognormal distribution 
## multiplying instead of summing the random variables together

nvals = 10
nsims = 10000
## when nsims is big, log(X) is normally distributed, so the distribution of X is called lognormal.
## it stems from a version of the central limit theorem applied to many random variables multiplied together
chisqsims = np.empty(nsims)

for i in range(nsims):
	chisq = scipy.stats.chi2.rvs(1, size=nvals)  ## this chi2 dist has 1 dof
	chisqsims[i] = scipy.stats.gmean(chisq)
	
fig, ax1 = plt.subplots(1, 1, figsize=(8,8))
countschi, edges = np.histogram(chisqsims, bins=30, range=[np.min(chisqsims), np.max(chisqsims)], density=True)
mu = edges[np.argmax(countschi)]

ax1.plot(edges[:-1], countschi, lw=3, color="black", linestyle="steps-mid")

print scipy.stats.chi2.stats(1, moments='mvsk')

# sigma = 1
s = 1

## what is s???

# s = 1 / np.sqrt(nvals)
x = np.arange(0., 2., 0.01)
# x = np.linspace(scipy.stats.lognorm.ppf(0.01, s), scipy.stats.lognorm.ppf(0.99, s), 100)  ## percentiles
# print x
sc = 1.0 / np.exp(mu)
# print sc

# n = scipy.stats.lognorm(s, scale=sc )
# ax1.plot(x, n.pdf(x), lw=3)

ax1.set_title("Lognormal distribution from geom mean of chi2")
plt.show()







