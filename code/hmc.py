#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:47:14 2017

@author: EmileMathieu
"""
import numpy as np
from tools import LogNormPDF
import timeit
    
def HMC(XX, t, NumOfIterations=6000, BurnIn=1000, NumOfLeapFrogSteps=100, StepSize=0.1):
    """ HAMILTONIAN MONTE CARLO """
    
    N,D = XX.shape
    
    # Prior covariance scaliing factor
    alpha=100    
    
    # HMC Setup
    Mass=np.eye(D)
    
    Proposed = 0
    Accepted = 0
    
    # Set initial values of w
    w = np.zeros((D,1))
    wSaved = np.zeros((NumOfIterations-BurnIn,D))
    
    # Calculate joint log likelihood for current w
    LogPrior      = LogNormPDF(np.zeros((1,D)), w, alpha)
    f             = np.dot(XX, w)
    LogLikelihood = np.dot(f.T,t) - np.sum(np.log(1+np.exp(f))) #training likelihood
    CurrentLJL    = LogLikelihood + LogPrior
    
    InvMass = np.linalg.inv(Mass) # precompute mass inverse
    
    for IterationNum in range(NumOfIterations):
        
        # Sample momentum
        ProposedMomentum = np.dot(np.random.randn(1,D), Mass).T
        OriginalMomentum = ProposedMomentum.copy()
        
        wNew = w.copy()
        
        Proposed += 1
        
        RandomStep = int(np.ceil(np.random.rand()*NumOfLeapFrogSteps))
            
        # Perform leapfrog steps
        for StepNum in range(RandomStep):
            f = np.dot(XX, wNew)
            likelihood_grad = np.dot(XX.T, t - np.exp(f)/(1+np.exp(f))) - np.eye(D).dot(wNew) / alpha
            ProposedMomentum += StepSize / 2 * likelihood_grad
    
            if np.sum(np.isnan(ProposedMomentum)) > 0:
                break
            wNew += StepSize * np.dot(InvMass, ProposedMomentum)
    
            f = np.dot(XX, wNew)
            likelihood_grad = np.dot(XX.T, t - np.exp(f)/(1+np.exp(f))) - np.eye(D).dot(wNew) / alpha
            ProposedMomentum += StepSize / 2 * likelihood_grad
            
        LogPrior      = LogNormPDF(np.zeros((1,D)),wNew,alpha)
        f             = np.dot(XX, wNew)
        LogLikelihood = np.dot(f.T, t) - np.sum(np.log(1+np.exp(f))) #training likelihood
        ProposedLJL   = LogLikelihood + LogPrior
        
        ProposedH = -ProposedLJL + ProposedMomentum.T.dot(InvMass).dot(ProposedMomentum)/2
            
        # Calculate current H value
        CurrentH = -CurrentLJL + OriginalMomentum.T.dot(InvMass).dot(OriginalMomentum)/2
           
        # Accept according to ratio
        Ratio = -ProposedH + CurrentH
    
        if Ratio > 0 or (Ratio > np.log(np.random.rand())):
            CurrentLJL = ProposedLJL
            w = wNew
            Accepted += 1
    
        # Save samples if required
        if IterationNum > BurnIn:
            wSaved[IterationNum-BurnIn, :] = w.T
        elif np.mod(IterationNum,50) == 0:
            print('{} iterations completed.'.format(IterationNum))
            print('Acceptance: {}'.format(Accepted/Proposed))
            Accepted = 0
            Proposed = 0
    
        # Start timer after burn-in
        if IterationNum == BurnIn:
            print('Burn-in complete, now drawing posterior samples.')
            start = timeit.default_timer()
    
    TimeTaken = timeit.default_timer() - start
    print('Time drawing posterior: {}'.format(TimeTaken))
    
    return wSaved, TimeTaken