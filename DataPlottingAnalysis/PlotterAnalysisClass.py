# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 12:57:18 2025

@author: 13ccu
"""

#IMPORTANT
#PLEASE READ
#This version is up to data as of: 03/03/2025

#This code uses a python binding of ODRPACK95 as its fitting method. As of 03/03/2025
#there is no implementation of ODRPACK95 in scipy. This version of ODRPACK allows constraints
#to be imposed on fit parameter values in addition to accounting for independent variable errors.
#This may not be the best fitting method and it may change in the future. Tom has used MCMC which
#may be worth considering in the future.

#The GitHub repo for the python binding of ODRPACK95 can be found here:
#https://github.com/HugoMVale/odrpack95
#The home page for the binding, which inclues API and documentation, is:
#https://pypi.org/project/odrpack/
#The discussion on replacing the current version of ODRPACK in scipy can be found here:
#https://github.com/scipy/scipy/issues/7107

#In order to install odrpack locally on your machine you can use the command:
#pip install odrpack
#I have found that I need to use the anaconda terminal in order for this to work.

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
from odrpack import odr
from scipy import linalg
from statsmodels.stats import stattools #Used for calculating Durbin-Watson statistic


class data_plotter():
    '''Given input data of one independent and one dependent variable with correspoding
    uncertainties will both plot data and also attempt to fit data using given fit function
    and ODRPACK95.'''
    
    def fitter(x, y, xerr, yerr, initial_guess, fit_func, lower_bound, upper_bound):
        '''Fits data to given fit function using ODRPACK95 which allows both parameter constraints and 
        accounts for errors in both dependent and independent variables. Plot of original data with
        determined fit also produced.'''
        
        #Use ODRPACK95 binding for fitting and use pandas for generality of code written
        sol = odr(fit_func, y, x, beta0 = initial_guess, lower = lower_bound, upper = upper_bound,
                  we = yerr, wd = xerr)
        parameter_estimates = sol.beta
        parameter_uncertainty_estimates = sol.sd_beta
        cov_matrix = sol.cov_beta
        iteration_number = sol.niter
        residual_sum_squares = sol.sum_square_eps
        sol.info
        sol.stopreason
        
        return parameter_estimates, parameter_uncertainty_estimates, cov_matrix,\
                iteration_number, residual_sum_squares
    
    def fit_func_linear(x, m, c):
        '''Fitting function of linear form. y = mx + c, where m is the gradient and c is the intercept.'''
        return m * x + c
    
    def fit_func_gaussian(x, A, mean, sigma, c):
        '''Fitting function of gaussian form. Here A is the amplitude and c is the vertical 
        offset of the Gaussian.Same definition as Wolfram:
        https://mathworld.wolfram.com/GaussianFunction.html'''
        return A * 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mean)**2)/(2 * sigma**2)) + c
        
    def fit_func_lorentzian(x, A, mean, gamma, c):
        '''Fitting function of Lorentzian form. Here A is the amplitude and c is the vertical
        offset of the Lorentzian with FWHM of gamma. Same definition as Wolfram:
        https://mathworld.wolfram.com/LorentzianFunction.html'''
        return A * 1/np.pi * ((gamma/2)/((x - mean)**2 + (gamma/2)**2)) + c
        
    def fit_func_pseudo_voigt(x, epsilon, A_1, A_2, sigma, gamma, mean_1, mean_2, c_1, c_2):
        '''Fitting function of pseudo-voigt form. Pseudo-voigt means weighted sum of Gaussian 
        ad Lorentzian lineshapes. This is a useful analytic approximation of the convolution
        between a Lorentzian and a Gaussian that a Voigt profile constitutes. If the fit struggles
        it might be useful to give the Lorentzian and Gaussian the same mean to prevent funky
        fitting from happening.'''
        return epsilon * A_1 * 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mean_1)**2)/(2 * sigma**2)) + c_1 + \
            (1 - epsilon) * A_2 * 1/np.pi * ((gamma/2)/((x - mean_2)**2 + (gamma/2)**2)) + c_2
        
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
        standard deviations. These normalised residuals should follow a standard normal 
        distribution.'''
        
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
        
        #Determine proportion of normalised residuals within 1, 2 and 3 sigma of 0
        counter_1 = 0 #Keeps track of how many data points have normalised residuals within 1 sigma of 0
        counter_2 = 0 #Keeps track of how many data points have normalised residuals within 2 sigma of 0
        counter_3 = 0 #Keeps track of how many data points have normalised residuals within 3 sigma of 0
        num_norm_residuals = np.size(normalised_residuals)
        for i in range(num_norm_residuals):
            if abs(normalised_residuals[i]) <= 1:
                counter_1 += 1
            elif abs(normalised_residuals[i]) <= 2:
                counter_2 += 1
            elif abs(normalised_residuals[i]) <= 3:
                counter_3 += 1
            else:
                counter_1 = counter_1
                counter_2 = counter_2
                counter_3 = counter_3
            proportion_1 = counter_1/num_norm_residuals
            print("The proportion of data points within 1 sigma of 0 is:", proportion_1)
            proportion_2 = counter_2/num_norm_residuals
            print("The proportion of data points within 2 sigma of 0 is:", proportion_2)
            proportion_3 = counter_3/num_norm_residuals
            print("The proportion of data points within 3 sigma of 0 is:", proportion_3)
                
        
        #Plot histogram of normalised residuals and fit to gaussian with mean 0 and std of 1
        
        return normalised_residuals 
        
        
    def reduced_chi_squared(RSS, num_degree_freedom):
        '''Computes the reduced chi squared given the RSS and number of degrees of freedom.
            This function also creates contour plots for the different parameters of 
            the fit applied to the data.'''
            
        print("The RSS of this fit is:", RSS)
        
        #Calculate reduced chi squared
        print("The number of degrees of freedom in this problem is:", num_degree_freedom)
        reduced_chi_squared = RSS/num_degree_freedom
        print("The reduced chi squared associated with this fit is:", reduced_chi_squared)
        
        #Contour plots of variation of fit parameters effect on chi squared
        
        #CONTOUR PLOTS HERE
        
        
        return reduced_chi_squared
        
    def durbin_watson(norm_residuals):
        '''Given the normalised residuals of the fit this function computes the Durbin Watson
        Statistic associated with the fit and creates lag plots of normalised residuals'''
        
        #Computing Durbin Watson
        durbin_watson = stattools.durbin_watson(norm_residuals)
        print("The Durbin Watson Statistic associated with this fit is:", durbin_watson)
        
        #Create lag plots of normalised residuals
        for i in range(np.size(norm_residuals)):
            plt.scatter(norm_residuals[i - 1], norm_residuals[i], color = 'blue')

        plt.xlabel("R$_{\mathrm{i - 1}}$")
        plt.ylabel("R$_\mathrm{i}$")
        plt.title("Lag Plot of Normalised Residuals")
        plt.show()
        
        
    def correlation_matrix(cov_fit_matrix):
        '''Given the covariance matrix from the fit applied in data_plotter() both the curvature
        and correlation matrix will be determined.'''
        
        #Covariance matrix as determined by fit
        covariance_matrix = cov_fit_matrix
        print("The covariance matrix associated with this fit is:", covariance_matrix)
        
        #Curvature matrix determined from covariance matrix
        curvature_matrix = linalg.inv(covariance_matrix)
        print("The curvature matrix associated with this fit is:", curvature_matrix)
        
        #Correlation matrix is determined element by element from the covariance matrix
        length_covariance_matrix = np.size(covariance_matrix, axis = 0)
        width_covariance_matrix = np.size(covariance_matrix, axis = 1)
        correlation_matrix = np.zeros([width_covariance_matrix, length_covariance_matrix])
        for i in range(width_covariance_matrix):
            for j in range(length_covariance_matrix):
                correlation_matrix[i, j] = covariance_matrix[i, j]/np.sqrt(covariance_matrix[i, i] * covariance_matrix[j, j])
        print("The correlation matrix associated with this fit is:", correlation_matrix)
        return covariance_matrix, curvature_matrix, correlation_matrix
        
        
        
        
    

