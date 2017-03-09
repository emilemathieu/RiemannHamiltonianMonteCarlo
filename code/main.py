#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:41:45 2017

@author: EmileMathieu
"""
import numpy as np
from hmc import HMC
from metropolis import AMH
from rmhmc import RMHMC
from gibbs_sampler import auxiliary_gibbs
from tools import nextpow2, ac, CalculateESS
#%% Load and preprocess data
X = np.loadtxt(open("data/australian.csv", "rb"), delimiter=",")
t = np.reshape(X[:,-1], (-1,1))
X = np.delete(X, -1, 1)
N = X.shape[0]

X = (X - np.tile(np.mean(X, axis=0), (N,1))) / np.tile(np.std(X, axis=0), (N, 1))

# Create Polynomial Basis
XX = np.ones((N,1))
XX = np.hstack((XX,X))

betaPosterior, TimeTaken = HMC(XX, t)
#betaPosterior = auxiliary_gibbs(XX, t)
#import pdb; pdb.set_trace()

#%%
#%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(range(betaPosterior.shape[0]), betaPosterior, linewidth=0.2)

#%%
#ACF = ac(betaPosterior[:,0], len(betaPosterior[:,0])-1)
#plt.plot(range(len(betaPosterior[:,0])-1), ACF)

#%%
print('ESS')
ESS = CalculateESS(betaPosterior, betaPosterior.shape[0]-1)

print('Min', np.min(ESS))
print('Median', np.median(ESS))
print('Mean', np.mean(ESS))
print('Max', np.max(ESS))
print('Time', TimeTaken)

print('Time per Min ESS:', round(TimeTaken / np.min(ESS), 6))