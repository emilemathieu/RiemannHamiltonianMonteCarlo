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
