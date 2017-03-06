#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:37:02 2017

@author: EmileMathieu
"""
#! cd /Users/EmileMathieu/Desktop/MVA/S2/Computational\ Statistics/Project/code

import numpy as np
from tools import LogNormPDF
import timeit

def AMH(XX, t, NumOfIterations=10000, BurnIn=5000):
    """ ADAPTIVE METROPOLIS HASTING """

    N,D = XX.shape

    # Prior covariance scaliing factor
    alpha=100 
    
    # Metropolis Setup
    ProposalSD = np.ones((D,1))
    
    Proposed = np.zeros((D,1))
    Accepted = np.zeros((D,1))
    
    # Set initial values of w
    w = np.zeros((D,1))
    wSaved = np.zeros((NumOfIterations-BurnIn,D))
    
    # Calculate joint log likelihood for current w
    LogPrior      = LogNormPDF(np.zeros((1,D)), w, alpha)
    f             = np.dot(XX,w)
    LogLikelihood = np.dot(f.T,t) - np.sum(np.log(1+np.exp(f))) #training likelihood
    CurrentLJL    = LogLikelihood + LogPrior
    
    for IterationNum in range(NumOfIterations):

        # For each w do metropolis step
        
        for d in range(D):
            wNew = w.copy()
            wNew[d] += np.random.normal()*ProposalSD[d]
        
            Proposed[d] += 1
            
            LogPrior      = LogNormPDF(np.zeros((1,D)),wNew,alpha)
            f             = np.dot(XX, wNew)
            LogLikelihood = np.dot(f.T, t) - np.sum(np.log(1+np.exp(f))) #training likelihood
            ProposedLJL   = LogLikelihood + LogPrior
        
            # Accept according to ratio
            Ratio = ProposedLJL - CurrentLJL
    
            if Ratio > 0 or (Ratio > np.log(np.random.rand())):
                CurrentLJL = ProposedLJL
                w = wNew
                Accepted[d] += 1
    
        # Save samples if required
        if IterationNum > BurnIn:
            wSaved[IterationNum-BurnIn, :] = w.T
        
        # Adjust sd every so often
        if np.mod(IterationNum,100) == 0 and IterationNum < BurnIn:
            
            if np.mod(IterationNum,1000) == 0:
                print('{} iterations completed.'.format(IterationNum))
            
            AcceptanceRatio = np.zeros((D, 1))
            for d in range(D):
                AcceptanceRatio[d] = Accepted[d]/Proposed[d]
                
                if AcceptanceRatio[d] > 0.5:
                    ProposalSD[d] *= 1.2
                elif AcceptanceRatio[d] < 0.2:
                    ProposalSD[d] *= 0.8
    
            # Print acceptance ratios
            #print(AcceptanceRatio)
            
            # Reset counters
            Accepted = np.zeros((D,1))
            Proposed = np.zeros((D,1))
    
        # Start timer after burn-in
        if IterationNum == BurnIn:
            print('Burn-in complete, now drawing posterior samples.')
            start = timeit.default_timer()
    
    TimeTaken = timeit.default_timer() - start
    print('Time drawing posterior: {}'.format(TimeTaken))

    return wSaved, TimeTaken