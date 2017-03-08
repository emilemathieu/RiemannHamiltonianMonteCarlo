#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:55:11 2017

@author: EmileMathieu
"""

import numpy as np
from tools import LogNormPDF
import timeit


X = np.loadtxt(open("data/australian.csv", "rb"), delimiter=",")
t = np.reshape(X[:,-1], (-1,1))
X = np.delete(X, -1, 1)
N = X.shape[0]

# Standardise Data
X = (X - np.tile(np.mean(X,axis=0), (N,1))) / np.tile(np.std(X,axis=0), (N,1))

# Create Polynomial Basis
XX = np.ones((N,1))
XX = np.hstack((XX,X))
#def RMHMC(XX, t, NumOfIterations=6, BurnIn=1, NumOfLeapFrogSteps=6, StepSize=0.5, NumOfNewtonSteps=4):
NumOfIterations=6000
BurnIn=1000
NumOfLeapFrogSteps=6
StepSize=0.5
NumOfNewtonSteps=4
""" RIEMANNIAN HAMILTONIAN MONTE CARLO """
 
N,D = XX.shape

# Prior covariance scaliing factor
alpha=100    

# HMC Setup

Proposed = 0
Accepted = 0

# Set initial values of w
w = np.ones((D,1))*1e-3
wSaved = np.empty((NumOfIterations-BurnIn,D))

# Calculate joint log likelihood for current w
LogPrior      = LogNormPDF(np.zeros((1,D)), w, alpha)
f             = np.dot(XX, w)
LogLikelihood = np.dot(f.T,t) - np.sum(np.log(1+np.exp(f))) #training likelihood
CurrentLJL    = LogLikelihood + LogPrior


for IterationNum in range(NumOfIterations):
    
    if np.mod(IterationNum+1,50) == 0:
        print('{} iterations completed.'.format(IterationNum+1))
        print('Acceptance: {}'.format(Accepted/Proposed))
        Accepted = 0
        Proposed = 0
    
    wNew = w.copy()
    Proposed += 1

    # Calculate G
    f = XX.dot(wNew)
    p = 1 / (1 + np.exp(-f))
    Lambda = np.diag((p * (np.ones((N,1)) - p))[:,0])
    G = XX.T.dot(Lambda).dot(XX) + np.eye(D) / alpha
    InvG = np.linalg.inv(G)
    OriginalG = G.copy()
    OriginalCholG = np.linalg.cholesky(G)
    OriginalInvG = InvG.copy()

    # Calculate the partial derivatives dG/dw
    InvGdG = np.empty((D,D,D))
    TraceInvGdG = np.empty((D,1))
    for d in range(D):
        V = np.diag(((np.ones((N,1)) - 2*p) * XX[:,d].reshape(-1,1))[:,0])
        GDeriv = XX.T.dot(Lambda).dot(V).dot(XX)
        InvGdG[d, :, :] = InvG.dot(GDeriv)
        TraceInvGdG[d] = np.trace(InvGdG[d, :, :])
    
    # Sample momentum
    ProposedMomentum = np.dot(np.random.randn(1,D), OriginalCholG).T
    OriginalMomentum = ProposedMomentum.copy()
    
    RandomStep = int(np.ceil(np.random.rand()*NumOfLeapFrogSteps))
    if (np.random.randn() > 0.5):
        TimeStep = 1
    else:
        TimeStep = -1

    # Perform leapfrog steps
    for StepNum in range(RandomStep):
        
        # Update momentum (Multiple fixed point iteration)
        f = XX.dot(wNew)
        likelihood_grad = np.dot(XX.T, t - np.exp(f)/(1+np.exp(f))) - np.eye(D).dot(wNew) / alpha
        
        PM = ProposedMomentum.copy()
        for FixedIter in range(NumOfNewtonSteps):
            InvGMomentum = InvG.dot(PM)
            LastTerm = np.empty((D, 1))
            for d in range(D):
                LastTerm[d]  = 0.5*PM.T.dot(InvGdG[d]).dot(InvGMomentum)
            PM = ProposedMomentum + TimeStep*StepSize/2 * (likelihood_grad - 0.5*TraceInvGdG + LastTerm)
        ProposedMomentum = PM

        # Update w parameters (Multiple fixed point iteration)
        OriginalInvGMomentum = np.linalg.inv(G).dot(ProposedMomentum)
        Pw = wNew.copy()
        for FixedIter in range(NumOfNewtonSteps):
            f = XX.dot(Pw)
            p = 1 / (1 + np.exp(-f))
            Lambda = np.diag((p * (np.ones((N,1)) - p))[:,0])
            G = XX.T.dot(Lambda).dot(XX) + np.eye(D) / alpha  
        
            InvGMomentum  = np.linalg.inv(G).dot(ProposedMomentum)
            Pw = wNew + TimeStep*StepSize/2 * (OriginalInvGMomentum + InvGMomentum)
        wNew = Pw

        # Update momentum
        InvG = np.linalg.inv(G)

        f = XX.dot(wNew)
        likelihood_grad = np.dot(XX.T, t - np.exp(f)/(1+np.exp(f))) - np.eye(D).dot(wNew) / alpha
        
        InvGdG = np.empty((D,D,D))
        TraceInvGdG = np.empty((D,1))
        for d in range(D):
            V = np.diag(((np.ones((N,1)) - 2*p) * XX[:,d].reshape(-1,1))[:,0])
            GDeriv = XX.T.dot(Lambda).dot(V).dot(XX)
            InvGdG[d] = InvG.dot(GDeriv)
            TraceInvGdG[d] = np.trace(InvGdG[d, :, :])
            
        InvGMomentum = InvG.dot(PM)
        LastTerm = np.empty((D, 1))
        for d in range(D):
            LastTerm[d]  = 0.5*PM.T.dot(InvGdG[d]).dot(InvGMomentum)
        
        ProposedMomentum += TimeStep*StepSize/2 * (likelihood_grad - 0.5*TraceInvGdG + LastTerm)
        
    LogPrior      = LogNormPDF(np.zeros((1,D)),wNew,alpha)
    f             = np.dot(XX, wNew)
    LogLikelihood = np.dot(f.T, t) - np.sum(np.log(1+np.exp(f))) #training likelihood
    ProposedLJL   = LogLikelihood + LogPrior
    
    ProposedLogDet = 0.5*(np.log(2) + D*np.log(np.pi) + 2*np.sum(np.log(np.diag(np.linalg.cholesky(G)))))
    ProposedH = -ProposedLJL + ProposedLogDet + ProposedMomentum.T.dot(InvG).dot(ProposedMomentum)/2
        
    # Calculate current H value
    CurrentLogDet = 0.5*(np.log(2) + D*np.log(np.pi) + 2*np.sum(np.log(np.diag(OriginalCholG))));
    CurrentH = -CurrentLJL + CurrentLogDet + OriginalMomentum.T.dot(OriginalInvG).dot(OriginalMomentum)/2

    # Accept according to ratio
    Ratio = -ProposedH + CurrentH

    if Ratio > 0 or (Ratio > np.log(np.random.rand())):
        CurrentLJL = ProposedLJL
        w = wNew
        Accepted += 1
        #print('Accepted')
    #else:
        #print('Rejected')

    # Save samples if required
    if IterationNum > BurnIn:
        wSaved[IterationNum-BurnIn, :] = w.T

    # Start timer after burn-in
    if IterationNum == BurnIn:
        print('Burn-in complete, now drawing posterior samples.')
        start = timeit.default_timer()

TimeTaken = timeit.default_timer() - start
print('Time drawing posterior: {}'.format(TimeTaken))

#return wSaved, TimeTaken