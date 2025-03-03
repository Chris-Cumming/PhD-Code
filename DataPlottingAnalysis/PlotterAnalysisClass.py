# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 12:57:18 2025

@author: 13ccu
"""

'''Current version is up to data as of: 03/03/2025 '''

#Import required modules

import numpy as np
import matplotlib.pyplot as plt


class data_plotter():
    '''Given input data of one independent and one dependent variable with correspoding
    uncertainties will both plot data and also attempt to fit data using given fit function
    and ODRPACK95.'''
    
    def fitter():
        '''Fits data to given fit function using ODRPACK95 which allows both parameter constraints and 
        accounts for errors in both dependent and independent variables. Plot of original data with
        determined fit also produced.'''
    
    
    #Add fitting functions over time
    #Current list of fitting functions available to parse to fitter() is:
        #linear fit
        #gaussian
        #lorentzian
        #pseudo-voigt
        #polynomial of order 2
    
    def fit_func_linear(x, m, c):
        '''Fitting function of linear form. y = mx + c, where m is the gradient and c is the intercept.'''
        return m * x + c
    
    def fit_func_gaussian():
        '''Fitting function of gaussian form.'''
        
        
    def fit_func_lorentzian():
        '''Fitting function of Lorentzian form.'''
        
    def fit_func_pseudo_voigt():
        '''Fitting function of pseudo-voigt form'''
        
    def poly2(x, a, b, c):
        '''Fitting function of polynomial of order 2: y = ax^2 + bx + c.'''
        return a * x**2 + b * x + c
    
    
    
    
    
class data_analysis():
    '''Given input data and estimated values from fit determined in data_plotter() will compute various
    statistical parameters and plots associated with quality of fit.'''
    
    def chi_squared():
        '''Computes the residual sum of squares (RSS) of fit. This function also creates 
        contour plots for the different parameters of the fit applied to the data.'''
        #Discrete RSS and continuous RSS are calculated differently.
        #print("The residual sum of squares (RSS) or chi squared of this fit is", chi_squared)
        
        
    def reduced_chi_squared():
        '''Computes the reduced chi squared given the RSS and degrees of freedom'''
        
        #print("The number of degrees of freedom in this problem is:", degrees of freedom)
        #reduced_chi_squared = 
        #print("The reduced chi squared associated with this fit is:", reduced_chi_squared)
        
        
    def normalised_residuals():
        '''Computes the residuals of the fit and then normalises them, the associated plot
        is then also provided along with the proportion of data points within 1, 2 and 3 
        standard deviations.'''
        #residuals = 
        #normalised_residuals = 
        #Plot normalised residuals
        #Determine proportion of normalised residuals within 1, 2 and 3 of 0
        #Plot histogram of normalised residuals and fit to gaussian with mean 0 and std of 1
        
        
    def durbin_watson():
        '''Computes the Durbin Watson Statistic associated with fit and creates lag plots
        of normalised residuals'''
        #durbin_watson = 
        #print(durbin_watson)
        #Create lag plots of normalised residuals
        #Determine proportion of normalised residuals within 1, 2 and 3 of 0 on lag plots as wwell
        
        
        
    def correlation_matrix():
        '''Given the covariance matrix from the fit applied in data_plotter() both the curvature
        and correlation matrix will be determined.'''
        #covariance_matrix = 
        #print(covariance_matrix)
        #curvature_matrix = 
        #print(curvature_matrix)
        #correlation_matrix = 
        #print(correlation_matrix)
        
        
        
        
        
    

