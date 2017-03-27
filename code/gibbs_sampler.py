#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:40:12 2017

@author: KimiaNadjahi
"""

import numpy as np
import timeit
import scipy.stats as stats


def rightmost_interval(U, Lambda):
    Z = 1
    X = np.exp(-0.5*Lambda)
    j = 0
    while 1:
        j += 1
        Z -= (j+1)**2 * X**((j+1)**2 - 1)
        if Z > U:
            return 1
        j += 1
        Z += (j+1)**2 * X**((j+1)**2 - 1)
        if Z < U:
            return 0


def leftmost_interval(U, Lambda):
    H = 0.5*np.log(2) + 2.5*np.log(np.pi) - 2.5*np.log(Lambda) - \
        np.pi**2 / (2*Lambda) + 0.5*Lambda
    logU = np.log(U)
    Z = 1
    X = np.exp((-np.pi**2) / (2*Lambda))
    K = Lambda / (np.pi**2)
    j = 0
    while 1:
        j += 1
        Z -= K*X**(j**2 - 1)
        logZ = np.log(Z) if Z > 0 else np.log(-Z)
        if H + np.log(Z) > logU:
            return 1
        j += 1
        Z += (j+1)**2 * X**((j+1)**2 - 1)
        logZ = np.log(Z) if Z > 0 else np.log(-Z)
        if H + np.log(Z) < logU:
            return 0


def mixing_weights_sampling(r2):
    """
    Drawing a sample of mixing weights using rejection sampling with GIG
    """
    r = np.sqrt(r2)
    OK = False
    while not OK:
        Y = np.random.normal()
        Y = Y*Y
        Y = 1 + (Y - np.sqrt(Y * (4*r + Y))) / (2*r)
        U = np.random.uniform()
        if U <= 1/(1+Y):
            Lambda = r/Y
        else:
            Lambda = r*Y
        U = np.random.uniform()
        if Lambda > 4/3:
            OK = rightmost_interval(U, Lambda)
        else:
            OK = leftmost_interval(U, Lambda)
    return Lambda


def auxiliary_gibbs(XX, t, v=100, max_iter=10000, burn_in=5000):
    """
    Auxiliary variable Gibbs sampler
    """
    print("--- Initialization...")
    N, D = XX.shape
    mix_weights = np.ones(N)
    beta_saved = np.zeros((max_iter-burn_in, D))
    # Truncated normal distribution
    t_ones = np.where(t == 1)[0]
    t_zeros = np.where(t == 0)[0]
    Z = np.zeros(N)
    mu, sigma = 0, 1
    for a in t_ones:
        lower, upper = 0, np.inf
        Z[a] = stats.truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    for a in t_zeros:
        lower, upper = -np.inf, 0
        Z[a] = stats.truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    print("--- Initialization: done. Iterating...")
    # Iterating
    for i in range(max_iter):
        if i % 100 == 0:
            print("Iteration %d" % i)
        # Timing after burn in
        if i == burn_in:
            start = timeit.default_timer()
        V = np.linalg.inv(XX.T.dot(np.diag(1/mix_weights)).dot(XX) + np.identity(D)*1/v)
        L = np.linalg.cholesky(V)
        S = V.dot(XX.T)
        B = S.dot(np.diag(1/mix_weights)).dot(Z)
        W = np.zeros(N)
        H = np.zeros(N)
        # Updating Z and B
        for j in range(N):
            z_old = Z[j]
            H[j] = XX[j, :].dot(S[:, j])
            W[j] = H[j] / (mix_weights[j] - H[j])
            m = XX[j].dot(B)
            m -= W[j] * (Z[j] - m)
            q = mix_weights[j] * (W[j] + 1)
            if t[j] == 1:
                lower, upper = 0, np.inf
                std = np.sqrt(q)
                Z[j] = stats.truncnorm.rvs(
                    (lower - m) / std, (upper - m) / std, loc=m, scale=std)
            else:
                lower, upper = -np.inf, 0
                std = np.sqrt(q)
                Z[j] = stats.truncnorm.rvs(
                    (lower - m) / std, (upper - m) / std, loc=m, scale=std)
            B += (Z[j] - z_old) / mix_weights[j] * S[:, j]
        # Drawing new values of beta
        T = np.random.multivariate_normal(np.zeros(D), np.identity(D))
        beta = B + L.dot(T)
        if i >= burn_in:
            beta_saved[i - burn_in] = beta
        # Sampling new mixing weights
        for j in range(N):
            r = Z[j] - XX[j, :].dot(beta)
            mix_weights[j] = mixing_weights_sampling(r**2)  # to time
    print("--- Iterating: done.")
    time = timeit.default_timer() - start
    print("--- Auxiliary Variable Gibbs Sampler finished in {}".format(time))
    return beta_saved, time
