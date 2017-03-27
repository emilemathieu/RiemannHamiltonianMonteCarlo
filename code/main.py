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
from iwls import iwls


if __name__ == '__main__':
    #%% Load and preprocess data
    dataset_name = 'australian'

    if dataset_name == 'heart':
        X = np.loadtxt(open("data/heart.csv", "rb"), delimiter=",")
        t = np.reshape(X[:, -1], (-1, 1))
        # Replacing classes 1 (resp. 2) by label 0 (resp. 1)
        t[t == 1] = 0
        t[t == 2] = 1
    elif dataset_name == 'australian':
        X = np.loadtxt(open("data/australian.csv", "rb"), delimiter=",")
        t = np.reshape(X[:, -1], (-1, 1))
    else:
        print("Error! Dataset must be 'australian' or 'heart'.")

    X = np.delete(X, -1, 1)
    N = X.shape[0]

    X = (X - np.tile(np.mean(X, axis=0), (N, 1))) / np.tile(np.std(X, axis=0), (N, 1))

    # Create Polynomial Basis
    XX = np.ones((N, 1))
    XX = np.hstack((XX, X))

    n_experiments = 10
    D = XX.shape[1]

    results_beta = np.zeros((n_experiments, 5000, D))
    results_time = np.zeros(n_experiments)
    for i in range(n_experiments):
        #results_beta[i], results_time[i] = iwls(XX, t)
        #results_beta[i], results_time[i] = auxiliary_gibbs(XX, t)
        #results_beta[i], results_time[i] = AMH(XX, t)
        #results_beta[i], results_time[i] = RMHMC(XX, t)
        results_beta[i], results_time[i] = HMC(XX, t)
    avg_beta_posterior = np.mean(results_beta, axis=0)
    avg_time_taken = np.mean(results_time)

    import pdb; pdb.set_trace()

    #%%
    #%matplotlib inline

    import matplotlib.pyplot as plt
    plt.plot(range(avg_beta_posterior.shape[0]), avg_beta_posterior, linewidth=0.2)

    #%%
    ACF = ac(avg_beta_posterior[:, 0], len(avg_beta_posterior[:, 0])-1)
    plt.plot(range(len(avg_beta_posterior[:, 0])-1), ACF)

    #%%
    print('ESS')
    ESS = CalculateESS(avg_beta_posterior, avg_beta_posterior.shape[0]-1)

    print('Min', np.min(ESS))
    print('Median', np.median(ESS))
    print('Mean', np.mean(ESS))
    print('Max', np.max(ESS))
    print('Time', avg_time_taken)

    print('Time per Min ESS:', round(avg_time_taken / np.min(ESS), 6))
