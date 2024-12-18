# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:35:27 2024

@author: xxvh83
"""

import numpy as np
import qutip as qp
import matplotlib.pyplot as plt

#QuTiP has 5 possible functions for various situations of calculating the correlation functions of operators.
#They can either compute correlation functions between 2 or 3 operators and then for either one time variable or two.

#For the case of <A(t) B(0)>, or the order reversed, we use qp.correlation_2op_1t
#This is equivalent to the initial state being the steady state, i.e we are computing the steady state correlation function.
#This is as a result of the quantum regression theorem.


#We will compute <A(t) B(0)> for a non-steady initial state later.
#The following example is for <x(t) x(0)> for a leaky cavity with three different relaxation rates of the cavity

times = np.linspace(0, 20, 200)
a = qp.destroy(10) #Annhilation operator for 10 level SHO
x = a.dag() + a #Operator we want to find autocorrelation of
H = a.dag() * a #Hamiltonian of system

#No initial state specified in this case - this, along with using 1 time variable,
#implies that we are dealing with a steady state initial solution. The code assumes the initial state
#is the steady state when given None.
correlation1 = qp.correlation_2op_1t(H, None, times, [np.sqrt(0.5) * a], x, x)
correlation2 = qp.correlation_2op_1t(H, None, times, [np.sqrt(1) * a], x, x)
correlation3 = qp.correlation_2op_1t(H, None, times, [np.sqrt(2) * a], x, x)

#print((correlation1))
#print(correlation2)
#print(correlation3)

#Only the real part of these correlation functions concern us - the imaginary parts are unphysical I think?

plt.plot(times, np.real(correlation1), label = '0.5')
plt.plot(times, np.real(correlation2), label = '1')
plt.plot(times, np.real(correlation3), label = '2')
plt.xlabel("Time (s)")
plt.ylabel("Correlation")
plt.title("Correlation of Leaky Cavity Assuming Initial Steady State Solution")
plt.legend()
plt.show()

#We now consider computing correlation functions for states that don't initially start in the steady state.
#QuTiP handles this with qp.correlation_2op_2t which returns a matrix with the values of the correlation given as a function
#of the two time coordinates.

times = np.linspace(0, 20, 200)
a = qp.destroy(10)
x = a.dag() + a
H = a.dag() * a

alpha = 2.5 #Eigenvalue of coherent state when operated on by annhilation operator
rho0 = qp.coherent_dm(10, alpha)
correlation = qp.correlation_2op_2t(H, rho0, times, times, [np.sqrt(0.25) * a], x, x)

plt.pcolor(np.real(correlation))
plt.xlabel("$t_1$")
plt.ylabel("$t_2")
plt.title("Autocorrelation for Leaky Cavity System")
plt.show()























