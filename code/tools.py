#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 00:38:00 2017

@author: EmileMathieu
"""
import numpy as np

def LogNormPDF(Values, Means, Variance):
    if (Values.shape[1] > 1):
        Values = Values.T
    D = Values.shape[0]
    return np.sum( -np.ones((D,1))*(0.5*np.log(2*np.pi*Variance)) - ((Values-Means)**2)/(2*(np.ones((D,1))*Variance)) )

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def ac(Series, nLag):
    Series = Series.flatten()
    nFFT =  (nextpow2(len(Series)) + 1)
    F    =  np.fft.fft(Series-np.mean(Series) , nFFT)
    F    =  F * np.conj(F)
    ACF  =  np.fft.ifft(F)
    ACF  =  ACF[0:nLag+1]         # Retain non-negative lags.
    ACF  =  ACF / ACF[0]     # Normalize.
    ACF  =  np.real(ACF)
    return ACF

def CalculateESS(Samples, MaxLag):
    MaxLag = int(MaxLag)
    NumOfSamples, NumOfParameters = Samples.shape

    # Calculate empirical autocovariance
    ACs = np.zeros((MaxLag+1, NumOfParameters))
    for i in range(NumOfParameters):
        ACs[:,i] = ac(Samples[:,i], MaxLag)
    
    halfMaxLag = int(np.floor((MaxLag+1)/2))
    Gamma    = np.zeros((halfMaxLag, NumOfParameters))
    MinGamma = np.zeros((halfMaxLag, NumOfParameters))

    # Calculate Gammas from the autocorrelations
    for i in range(NumOfParameters):
        
        # Add other Gammas
        for j in range(halfMaxLag):
            Gamma[j,i] = ACs[2*(j+1)-2,i] + ACs[2*(j+1)-1,i]
            
    # Calculate the initial monotone convergence estimator
    # -> Gamma[j,i] is min of preceding values
    for i in range(NumOfParameters):
        # Set initial min Gamma
        MinGamma[0,i] = Gamma[0,i]
        
        for j in range(1,halfMaxLag):
            MinGamma[j,i] = min(Gamma[j,i], MinGamma[j-1,i])
            Gamma[j,i] = MinGamma[j,i]
    
    MonoEst = np.zeros((NumOfParameters, 1))
    for i in range(NumOfParameters):
        # Get indices of all Gammas greater than 0
        PosGammas = np.nonzero(Gamma[:,i]>0)[0]
        # Sum over all positive Gammas
        MonoEst[i] = -ACs[0,i] + 2*np.sum(Gamma[0:len(PosGammas),i])
        
        # MonoEst cannot be less than 1 - fix for when lag 2 corrs < 0
        if MonoEst[i] < 1:
            MonoEst[i] = 1

    ESS = NumOfSamples / MonoEst
    return ESS