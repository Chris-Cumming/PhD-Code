# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:31:12 2024

@author: xxvh83
"""

#Load necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize

#Load in data
df = pd.read_csv("Z:\\Training\\TOPTICA DL PRO 780 Repump\\9-10-2024\\PowerCurrentThreshold.csv")
#print(df)


#Current in mA and power in mW
power_current_data = df.to_numpy()
#print(power_current_data)


#Current and power values in separate numpy arrays
current_data = power_current_data[:, 0]
power_data = power_current_data[:, 1]
power_error_data = power_current_data[:, 2]
#print(current_data)
#print(power_data)
#print(power_error_data)

#Define functions for fitting use curve_fit

def linear_regression(x, m, c):
    '''Define linear regression function we wish to fit to data. m is gradient and c is intercept'''
    return m*x + c

def tanh_func(x, A, s, c):
    '''Define tanh function we wish to fit to data. A is amplitude, s is scaling, c is offset. x is data values.'''
    return A * np.tanh(s*x) + c


#Remove data before lasing threshold so we can apply linear regression fit to remaining data.
new_current_data = current_data[2:]
new_power_data = power_data[2:]
new_power_error_data = power_error_data[2:]
#print(new_current_data)
#print(new_power_data)
#print(new_power_error_data)
#plt.plot(new_current_data, new_power_data)

'''Linear fit to data'''

#Now fit linear regression to remaining data
linear_popt, linear_pcov = optimize.curve_fit(linear_regression, new_current_data, new_power_data) #sigma = new_power_error_data
#popt is a 1D array containing the optimal values for the slope and intercept of the fit
#pcov is a 2D array the diagonals of which give the errors of the optimal values

#print(linear_popt)
#print(pcov)

#Extract parameters from linear regression fit
linear_optimal_slope = linear_popt[0]
#print(linear_optimal_slope)
linear_optimal_intercept = linear_popt[1]
#print(linear_optimal_intercept)
linear_optimal_slope_error = linear_pcov[0, 0]
#print(linear_optimal_slope_error)
linear_optimal_intercept_error = linear_pcov[1, 1]
#print(linear_optimal_intercept_error)

linear_lasing_threshold = abs(linear_optimal_intercept)/abs(linear_optimal_slope)
linear_lasing_threshold_error = abs(linear_optimal_intercept + linear_optimal_intercept_error)/(abs(linear_optimal_slope) - linear_optimal_slope_error) - linear_lasing_threshold
print(linear_lasing_threshold)
print(linear_lasing_threshold_error)

'''Tanh fit to data'''

#Now fit tanh functino to remaining data
tanh_popt, tanh_pcov = optimize.curve_fit(tanh_func, new_current_data, new_power_data) #sigma = new_power_error_data
#popt is a 1D array containing the optimal values for the parameters of the fit
#pcov is a 2D array the diagonals of which give the errors of the optimal values

#print(popt)
#curve_fit is unable to find covariances of fit - so no errors
#print(pcov)

#Extract parameters from linear regression fit
tanh_optimal_amplitude = tanh_popt[0]
#print(tanh_optimal_amplitude)
tanh_optimal_scaling = tanh_popt[1]
#print(tanh_optimal_scaling)
tanh_optimal_intercept = tanh_popt[2]
#print(tanh_optimal_intercept)
tanh_optimal_amplitude_error = tanh_pcov[0, 0]
tanh_optimal_scaling_error = tanh_pcov[1, 1]
tanh_optimal_intercept_error = tanh_pcov[2, 2]

'''Form lines of best fit from parameters determined'''

test_current_values = np.array([70, 80, 90, 110, 130, 150, 170, 190, 210, 230, 250])
#print(test_current_values)

#Linear regression
linear_test_power_values = linear_optimal_slope * test_current_values + linear_optimal_intercept
print(linear_test_power_values)
plt.plot(test_current_values, linear_test_power_values, 'r', label = "Linear Regression")


'''
#Tanh function
tanh_test_power_values = tanh_optimal_amplitude * np.tanh(tanh_optimal_scaling * test_current_values)
print(tanh_test_power_values)
plt.plot(test_current_values, tanh_test_power_values, 'g')
'''


#Plot of the fits and original data
plt.scatter(current_data, power_data, label = "Original Data")
plt.xlabel("Current (mA)")
plt.ylabel("Power (mW)")
plt.errorbar(current_data, power_data, yerr = power_error_data, ls = 'none')
plt.legend(loc = 'upper left')
plt.title("Output Power of Repump Laser")
plt.show()









