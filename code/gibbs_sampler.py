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


def mixing_weights_sampling(r):
    """
    Drawing a sample of mixing weights using rejection sampling with GIG
    """
    OK = False
    while not OK:
        Y = np.random.normal(0, 1)
        Y = Y**2
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


def auxiliary_gibbs(XX, t, v, beta, max_iter=6000, burn_in=5000):
    """
    Auxiliary variable Gibbs sampler
    """
    N, D = XX.shape
    mix_weights = np.identity(N)
    # Truncated normal distribution
    Z = np.zeros(N)
    t_ones = np.where(t == 1)
    t_zeros = np.where(t == 0)
    for ind in t_ones:
        Z[ind] = np.random.multivariate_normal(np.zeros(N), np.identity(N)) * (Z[ind] > 0)
    for ind in t_zeros:
        Z[ind] = np.random.multivariate_normal(np.zeros(N), np.identity(N)) * (Z[ind] <= 0)

    H = np.zeros(N)
    W = np.zeros(N)
    # Iterating
    for i in range(max_iter):
        mix_weights_inv = np.linalg.inv(mix_weights)
        V = np.linalg.inv(XX.T.dot(mix_weights_inv.dot(XX)) + np.identity(D)*1/v)
        L = np.linalg.cholesky(V)
        S = V.dot(XX.T)
        B = S.dot(mix_weights_inv.dot(Z))
        for j in range(N):
            z_old = Z[j]
            H[j] = XX[j, :].dot(S[:, j])
            W[j] = H[j] / (mix_weights[j, j] - H[j])
            m = XX[j, :].dot(B)
            m -= W[j]*(Z[j] - m)
            q = mix_weights[j, j] * (W[j] + 1)
            # Truncated normal distribution
            if t[j] == 1:
                Z[j] = np.random.multivariate_normal(m, q) * (Z[j] > 0)
            else:
                Z[j] = np.random.multivariate_normal(m, q) * (Z[j] <= 0)
            B += (Z[j] - z_old) / mix_weights[j, j] * S[:, j]
        p = L.shape[1]
        T = np.random.multivariate_normal(np.zeros(p), np.identity(p))
        beta = B + L.dot(T)  # save all betas?
        for j in range(N):
            r = Z[j] - XX[j, :]*beta
            mix_weights[j][j] = mixing_weights_sampling(r)
    return beta
