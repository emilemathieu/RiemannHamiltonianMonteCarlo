#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:40:12 2017

@author: KimiaNadjahi
"""

import numpy as np


def rightmost_interval(U, Lambda):
    Z = 1
    X = np.exp(-0.5*Lambda)
    j = 0
    while 1:
        j += 1
        Z -= (j+1)**2 * np.power(X, (j+1)**2 - 1)
        if Z > U:
            return 1
        j += 1
        Z += (j+1)**2 * np.power(X, (j+1)**2 - 1)
        if Z < U:
            return 0


def leftmost_interval(U, Lambda):
    H = 0.5*np.log(2) + 2.5*np.log(np.pi) - 2.5*np.log(Lambda) - \
        np.pi**2 / (2*Lambda) + 0.5*Lambda
    logU = np.log(U)
    Z = 1
    X = np.exp(-np.pi**2 / (2*Lambda))
    K = Lambda / np.pi**2
    j = 0
    while 1:
        j += 1
        Z -= K*np.power(X, j**2-1)
        if H+np.log(Z) > logU:
            return 1
        j += 1
        Z += (j+1)**2 * np.power(X, (j+1)**2 - 1)
        if H + np.log(Z) < logU:
            return 0


def mixing_weights_sampling(r2):
    """
    Drawing a sample of mixing weights using rejection sampling with GIG
    """
    r = np.sqrt(r2)
    OK = False
    while not OK:
        Y = (np.random.normal(0, 1))**2
        Y = 1 + (Y - np.sqrt(Y * (4*r + Y))) / (2*r)
        U = np.random.uniform(0, 1)
        if U <= 1/(1+Y):
            Lambda = r/Y
        else:
            Lambda = r*Y
        U = np.random.uniform(0, 1)
        if Lambda > 4/3:
            OK = rightmost_interval(U, Lambda)
        else:
            OK = leftmost_interval(U, Lambda)
    return Lambda


def truncated_normal_sampling(Z, t, mean, var):
    t_ones = np.where(t == 1)[0]
    t_zeros = np.where(t == 0)[0]
    for i in t_ones:
        Z[i] = np.random.normal(mean[i], var[i]) * (Z[i] > 0)
    for i in t_zeros:
        Z[i] = np.random.normal(mean[i], var[i]) * (Z[i] <= 0)
    return Z


def auxiliary_gibbs(XX, t, v=100, max_iter=6000, burn_in=5000):
    """
    Auxiliary variable Gibbs sampler
    """
    print("--- Initialization...")
    N, D = XX.shape
    mix_weights = np.identity(N)
    beta_saved = np.zeros((max_iter-burn_in, D))
    # Truncated normal distribution
    Z = np.zeros(N)
    # t_ones = np.where(t == 1)[0]
    # t_zeros = np.where(t == 0)[0]
    # for ind in t_ones:
    #     Z[ind] = np.random.normal(0, 1) * (Z[ind] > 0)
    # for ind in t_zeros:
    #     Z[ind] = np.random.normal(0, 1) * (Z[ind] <= 0)
    Z = truncated_normal_sampling(Z, t, np.zeros(N), np.ones(N))

    print("--- Initialization: done. Iterating...")
    # Iterating
    for i in range(max_iter):
        if i % 100 == 0:
            print("Iteration %d" % i)
        mix_weights_inv = np.linalg.inv(mix_weights)
        V = np.linalg.inv(XX.T.dot(mix_weights_inv.dot(XX)) + np.identity(D)*1/v)
        L = np.linalg.cholesky(V)
        S = V.dot(XX.T)
        B = S.dot(mix_weights_inv.dot(Z))
        # Updating Z and B
        # for j in range(N):
        #     z_old = Z[j]
        #     H[j] = XX[j, :].dot(S[:, j])
        #     W[j] = H[j] / (mix_weights[j, j] - H[j])
        #     m = XX[j, :].dot(B)
        #     m -= W[j]*(Z[j] - m)
        #     q = mix_weights[j, j] * (W[j] + 1)
        #     # Truncated normal distribution
        #     Z[j] = np.random.normal(m, q) * ((Z[j] > 0) * (t[j] == 1) +
        #                                      (Z[j] <= 0) * (t[j] == 0))
        #     B += (Z[j] - z_old) / mix_weights[j, j] * S[:, j]

        z_old = Z
        H = (XX.T*S).sum(axis=0)
        W = H / (np.diag(mix_weights) - H)
        m = XX.dot(B)
        m -= W*(Z - m)
        q = np.diag(mix_weights) * (W + np.ones(N))
        Z = truncated_normal_sampling(Z, t, m, q)
        B = ((Z - z_old) / (np.diag(mix_weights)*S)).sum(axis=1)

        p = L.shape[1]
        T = np.random.multivariate_normal(np.zeros(p), np.identity(p))
        # Drawing new values of beta
        beta = B + L.dot(T)
        if i > burn_in:
            beta_saved[i - burn_in] = beta
        # Sampling new mixing weights
        for j in range(N):
            r2 = (Z[j] - XX[j, :].dot(beta))**2
            mix_weights[j, j] = mixing_weights_sampling(r2)
    print("--- Iterating: done.")
    return beta_saved
