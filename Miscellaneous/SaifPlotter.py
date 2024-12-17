# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:56:38 2024

PLEASE make a copy of this file - do not edit the original source code unless you find a glaring issue or are adding an improvement. If you do modify the original,
please document your changes well.

This program is designed to plot a dataset, fit a user-defined fit function, and compute normalized residuals. The definition takes in quite a few parameters, which 
are fairly clear based on their names. Note that the definition plotWithResiduals will not actually display the plot - I have decided to make that step external
in case any additional features are wanted as well. Returned to the user are the fit parameters, covariance matrix, figure, and subplots, with axs[0] containing the data and fit and axs[1] containing 
the normalized residuals.

To use the code, import your CSV on line 65. Then, assign each column (like lines 67-70). Finally, modify the function's parameters in lines 72-83.
Any adjustment to the plot can be done at line 94 using the definitions from 87-91.

@author: Saif Salim
"""
'''
NOTE: Here we are accounting for the change in axes that occur for the different data sets. So in the lab frame, the z axis is the axis
which connects pump and pyramid; x is in the plane of the pyramid MOT and is the same as conventional x axis for a birds eye view of chamber
i.e positive x is in direction towards filter; the same is true for the y axis, i.e positive y is in the direction away from the optical bench
with cooling and repump setups.
'''
#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def plotWithResiduals(xData, yData, xLabel, yLabel, Title, yErr, fitFunction, ParamGuesses, xErr = None, fitOverhang = 1.02, residualBuffer = 1.3):
    # Setting the extra space for the fit
    if min(xData)>0: 
        xMinLim = min(xData)/fitOverhang
    else:
        xMinLim = min(xData)*fitOverhang
    if max(xData)>0: 
        xMaxLim = max(xData)*fitOverhang
    else:
        xMaxLim = max(xData)/fitOverhang

    # Define a Fit Function
    x_arr = np.linspace(int(xMinLim), int(xMaxLim), 1000)
    fitParams, pcov = curve_fit(fitFunction, xdata=xData, ydata=yData, p0=ParamGuesses, sigma=yErr, absolute_sigma=True)
    y_fit = fitFunction(x_arr, *fitParams)
    residuals = np.array(yData - fitFunction(xData, *fitParams))
    norm_res = residuals/yErr
    fit_err = np.sqrt(np.diag(pcov))

    print('The best-fit parameters for the theory function are:')
    print(fitParams) 
    print('The errors for the associated fit parameters are:')
    print(fit_err)
    
    # Visualise
    fig, axs = plt.subplots(1, sharex = 'col', figsize = (7, 5), height_ratios = [3, 1], constrained_layout = True)
    axs[0].errorbar(xData, yData, yerr=yErr, xerr = xErr, fmt='o', capsize=5, label='Data')
    #axs[0].plot(x_arr, y_fit, '--', label='Fit Function', zorder=10)

    axs[0].set(xlabel=xLabel, ylabel=yLabel, title = Title)
    axs[0].legend()
    
    #axs[1].scatter(xData, norm_res)
    # axs[1].hlines(y=0, xmin=xMinLim, xmax=xMaxLim, colors='k', alpha=.3)
    #axs[1].set(xlabel=xLabel, ylabel='Normalised Residuals')
    #axs[1].set_ylim(-max(abs(norm_res))*residualBuffer,max(abs(norm_res))*residualBuffer)
    return [fitParams, pcov, fig, axs]

#Define your fit function! A few have been provided
def Gaussian(x, A, x0, sigma):
    return A*np.exp(-(x-x0)**2/(2*(sigma**2)))
def Linear(x, m, b):
    return m*x + b
def Quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Import measurement data
data = pd.read_csv("Z:\\Training\\MOT\\Ion Pump\\10-12-2024\\IonPumpMagneticFieldVerticalDistance.csv").to_numpy() # Note - this import wants your csv to have column names. If there aren't any, import using numpy instead.

XData = data[:,0]
XErr = data[:, 1]
YData = data[:,4]
YErr = data[:,5]

returnedArguments = plotWithResiduals(xData=XData, 
                                      yData=YData, 
                                      xLabel='Vertical Distance (cm)', 
                                      yLabel='$B_x$ (Gauss)', 
                                      Title='Variation of Magnetic Field from Ion Pump', 
                                      yErr=YErr, 
                                      fitFunction=Quadratic, # Choose your fit function here 
                                      ParamGuesses=[1, 1, 1], 
                                      xErr = XErr,
                                      fitOverhang= 1.02, # Scales the x axis of the data plot, to see more of the fit function
                                      residualBuffer=1.3 # Scales the y axis of the residuals plot, ensure residuals aren't clipping the frame
                                      )

# Returned values expressed more meaningfully
fitParameters = returnedArguments[0]
covarianceMatrix = returnedArguments[1]
figure = returnedArguments[2]
DataSubPlot = returnedArguments[3][0]
ResidualsSubPlot = returnedArguments[3][1]

#Make any adjustments to the plot here


plt.show()