# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 12:57:18 2025

@author: 13ccu
"""

#IMPORTANT
#PLEASE READ
#This version is up to data as of: 03/03/2025

#This code uses a python binding of ODRPACK95 as its fitting method. As of 03/03/2025
#there is no implementation of ODRPACK in scipy. This version of ODRPACK allows constraints
#to be imposed on fit parameter values in addition to accounting for independent variable errors.
#This may not be the best fitting method and it may change in the future. Tom has used MCMC which
#may be worth considering in the future.

#Fitting functions will be added over time as and when they are needed.
#Current list of fitting functions available to parse to fitter() is:
    #linear fit
    #gaussian
    #lorentzian
    #pseudo-voigt
    #polynomial of order 2
    #polynomial of order 3

#Import required modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats import stattools #Used for calculating Durbin-Watson statistic


class data_plotter():
    '''Given input data of one independent and one dependent variable with correspoding
    uncertainties will both plot data and also attempt to fit data using given fit function
    and ODRPACK95.'''
    
    def fitter():
        '''Fits data to given fit function using ODRPACK95 which allows both parameter constraints and 
        accounts for errors in both dependent and independent variables. Plot of original data with
        determined fit also produced.'''
    
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
    
    def poly3(x, a, b, c, d):
        '''Fitting function of polynomial of order 3: y = ax^3 + bx^2 + cx + d.'''
        return a * x**3 + b * x**2 + c * x + d
    
    
    
class data_analysis():
    '''Given input data and estimated values from fit determined in data_plotter() will compute various
    statistical parameters and plots associated with quality of fit.'''
                
    def normalised_residuals(fit_data, original_data, original_data_yerr):
        '''Computes the residuals of the fit and then normalises them, the associated plot
        is then also provided along with the proportion of data points within 1, 2 and 3 
        standard deviations.'''
        #Compute residuals and normalised residuals
        residuals = fit_data - original_data
        normalised_residuals = residuals/original_data_yerr
        #Create scatter plot of normalised residuals
        plt.scatter(original_data, normalised_residuals)
        plt.xlabel("X Label") #Use heading of column from pandas to determine x labe
        plt.ylabel("Normalised Residuals Value")
        plt.xlim([np.min(original_data) - 1, np.max(original_data) + 1])
        plt.ylim([np.min(normalised_residuals) - 1, np.max(normalised_residuals) + 1])
        plt.show()
        #Determine proportion of normalised residuals within 1, 2 and 3 of 0
        #Plot histogram of normalised residuals and fit to gaussian with mean 0 and std of 1
        return normalised_residuals 
        
    def chi_squared(norm_residuals, observed_value, expected_value, continuous_marker):
        '''Computes the residual sum of squares (RSS) of fit. This function also creates 
        contour plots for the different parameters of the fit applied to the data.
        The parameter norm_residuals represents the normalised residuals calculated previously and
        is used to calculate RSS for continuous data; observed and expected value are used tor
        discrete data after binning. The continuous_marker parameter defines how the RSS is
        calculated, use TRUE for continuous data and FALSE for discrete (binned) data.'''
        #Discrete RSS and continuous RSS are calculated differently
        if continuous_marker == "TRUE":
            RSS = np.sum(norm_residuals**2)
            print("RSS calculated assuming continuous data set.")
        elif continuous_marker == "FALSE":
            RSS = np.sum(((observed_value - expected_value)**2)/expected_value)
            print("RSS calculated assuming discrete data set.")
        else:
            print("Error in chi_squared(). Unknown continuous marker parsed.")
        print("The residual sum of squares (RSS) or chi squared of this fit is", RSS)
        #Contour plots of variation of fit parameters effect on chi squared
        
        #CONTOUR PLOTS HERE
        
        return RSS
        
        
    def reduced_chi_squared(RSS, num_degree_freedom):
        '''Computes the reduced chi squared given the RSS and number of degrees of freedom.'''
        print("The number of degrees of freedom in this problem is:", num_degree_freedom)
        reduced_chi_squared = RSS/num_degree_freedom
        print("The reduced chi squared associated with this fit is:", reduced_chi_squared)
        return reduced_chi_squared
        
        
    def durbin_watson(norm_residuals):
        '''Given the normalised residuals of the fit this function computes the Durbin Watson
        Statistic associated with the fit and creates lag plots of normalised residuals'''
        #Computing Durbin Watson
        durbin_watson = stattools.durbin_watson(norm_residuals)
        print("The Durbin Watson Statistic associated with this fit is:", durbin_watson)
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
        
        
        
        
        
    

