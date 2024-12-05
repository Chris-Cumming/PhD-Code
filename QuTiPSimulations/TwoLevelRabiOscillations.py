# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:04:21 2024

@author: xxvh83
"""

import numpy as np
import qutip as qp
import matplotlib.pyplot as plt
'''
Here we determine the behaviour of a two level system. We do this with both sesolve
and mesolve. sesolve can only be used when neglecting dissipation of system. We begin by using sesolve
to determine the behaviour of a two level system in the absence of spontaneous emission. This results
in Rabi oscillations. We repeat this calculation using mesolve. We then use mesolve to determine the
behaviour of the system when accounting for spontaneous emission.
'''
#Define functions

def hamiltonian(rabi_freq, detuning_freq):
    '''Define hamiltonian of system. This form has had the RWA applied.
    See ALI notes for derivation'''
    H_rabi = 2 * np.pi * 0.5 * qp.sigmax() * rabi_freq
    H_det = - 2 * np.pi * 0.5 * qp.sigmaz() * detuning_freq
    H = H_rabi + H_det
    return H

def plots(x_values, y_values1, y_values2, title):
    '''Plot behaviour of system given time intervals (x_values) and populations of
    ground (y_values1) and excited (y_values2) states with solver used (title)'''
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Level Occupation")
    plt.plot(x_values, y_values1, label = 'Ground State')
    plt.plot(x_values, y_values2, label = 'Excited State')
    plt.legend(loc = 'upper right')
    plt.title(title)
    plt.show()
    return None

    

#Define parameters and Hamiltonian of system

dim_system = 2 #Dimension of Hilbert space of system being considered, 2 level system
occupied_state = 0 #Indicates which level of the system is initially occupied, 0 refers to ground state, 1 to first excited state, etc
unoccupied_state = 1 #Indicates the excited state level which is initially unoccupied at t = 0
rabi_freq = 1
detuning_freq = 0

#Define times to solve for to include 10 full Rabi oscillations
time_interval = 10 * 1/rabi_freq #Increase of decrease number of oscillations from 10

g = qp.fock(dim_system, occupied_state) #State vector of ground state which is the initial state of system
e = qp.fock(dim_system, unoccupied_state) #State vector of excited state of system
times = np.linspace(0, time_interval, 1000)

result_size = np.size(times)

H_total = hamiltonian(rabi_freq, detuning_freq) #Compute hamiltonian of system given system parameters

#Solve system dynamics using sesolver. This solves the TDSE given the hamiltonian and initial
#state of the system for the time interval provided. All useful data contained in result class.
result = qp.sesolve(H_total, g, times)

#Initialise arrays to store values for probability of state being occupied
ground_state_population = np.zeros(result_size)
excited_state_population = np.zeros(result_size)

system_evolution = result.states #Obtain values of state vectors for each time.

for i in range(result_size):
    state_evolution = system_evolution[i].full()
    ground_state_population[i] = abs(state_evolution[0])**2 #Magnitude of coefficient squared is probability
    excited_state_population[i] = abs(state_evolution[1])**2 #Magnitude of coefficient squared is probability


#Plot resulting behaviour of state vectors
plots(times, ground_state_population, excited_state_population, "sesolver")


'''Perform same calculation using mesolve instead of sesolve'''

#Solve with mesolve with no collapose operators - Linbladian is then equivalent to Liouville equation
#This is density matrix equivalent to TDSE which is used in sesolve.
#Have to provide a density matrix instead of state function.

initial_dm = qp.fock_dm(dim_system, occupied_state) #Define initial state, population is sum of diagonal.
times = np.linspace(0, time_interval, 1000) #Define times to include 4 full Rabi oscillations

#Define operators that we parse to mesolve which returns expectation value of operator at each time.

expectation_g = g * g.dag() #Operator corresponding to occupation of ground state
#print(expectation_g)
expectation_e = e * e.dag() #Operator corresponding to occupation of excited state
#print(expectation_e)

#Solve system dynamics using mesolver. This solves the Linbladian given the hamiltonian and initial
#state of the system for the time interval provided. All useful data contained in result class. Here
#we also parse the operators we want the expectation values of.
solution = qp.mesolve(H_total, initial_dm, tlist = times, e_ops = [expectation_g, expectation_e])


#Much simpler/cleaner way of seeing evolution of system compared to manually extracting coefficients of states
system_evolution_dm_ground = solution.expect[0] #Ground state population for each time
system_evolution_dm_excited = solution.expect[1] #Excited state population for each time


#Plot resutling behaviour of system
plots(times, system_evolution_dm_ground, system_evolution_dm_excited, "mesolver")

'''Now use mesolve to account for spontaneous emission in two level system.'''

#Create collapse operators that describe the process of spontaneous emission between
#ground and excited state. These are the Linblad (jump) operators which account for dissipation
#in system.

#Create a list to contain all of the operators for all possible dissipation channels.
c_ops_list = [] #Empty list
gamma = 1 #Determines rate of dissipation
c1 = -1 * np.sqrt(gamma) * g * e.dag() #'Lowering operator' for spontaneous emission
c_ops_list.append(c1)


#Solve system dynamics using mesolver. This solves the Linbladian given the hamiltonian and initial
#state of the system for the time interval provided. All useful data contained in result class. Here
#we also parse the operators we want the expectation values of as well as the collapse operators.
solution = qp.mesolve(H_total, initial_dm, tlist = times,c_ops = c_ops_list, e_ops = [expectation_g, expectation_e])

system_evolution_dm_ground = solution.expect[0]
system_evolution_dm_excited = solution.expect[1]

#Plot resulting behaviour of system
plots(times, system_evolution_dm_ground, system_evolution_dm_excited, "mesolver (Spontaneous Emission)")



