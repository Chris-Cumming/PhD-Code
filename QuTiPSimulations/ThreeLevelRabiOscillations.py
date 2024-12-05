# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:47:09 2024

@author: 13ccu
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qp

'''
Here we consider a three level system in the lambda configuration. This is analagous to the usual 
two level system except now there is a third level present just below the excited state. Here we
consider this state to be dark, meaning it can't be addressed by the laser and there is no natural
decay pathway once this state is occupied. However this state can be reached by spontaneous emission
from the excited state.
'''

def hamiltonian(rabi_freq, detuning_freq):
    '''Extension of two level system to include dark state. All extra terms must be
    zero since the laser doesn't address the dark state so no Rabi oscillation or detuning terms.'''
    H_rabi = 2 * np.pi * rabi_freq * np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])
    H_det = - 2 * np.pi * detuning_freq * np.array([[0.5, 0, 0], [0, -0.5, 0], [0, 0, 0]])
    H = H_rabi + H_det
    return H

rabi_freq = 1
detuning_freq = 0
H3 = hamiltonian(rabi_freq, detuning_freq) #Same as two level but with extra zero terms

three_level_hamiltonian = qp.Qobj(H3) #Allows hamiltonian to be inputted to QuTiP functions

dim_system = 3 #Dimension of Hilbert space being considered

#Define state functions energy levels of system. 
ground_state = qp.fock(dim_system, 0)
excited_state = qp.fock(dim_system, 1)
dark_state = qp.fock(dim_system, 2)

#Create operators that we want to know the expectation values of. These are the operators that
#return level populations.
expectation_g = ground_state * ground_state.dag()
expectation_e = excited_state * excited_state.dag()
expectation_d = dark_state * dark_state.dag()

#Initial state of system is the ground state - described by density matrix
initial_dm = qp.fock_dm(dim_system, 0)

#Define time duration of system
time_interval = 15 * 1/rabi_freq #Define times to solve Linbladian to include 15 'Rabi Oscillations'
times = np.linspace(0, time_interval, 1000)

#Define collapse operators of system
c_ops_list = []
gamma = 1 #Defines rate of decay
c1 = -1 * np.sqrt((gamma/2)/2) * ground_state * excited_state.dag() #Excited to ground decay pathway
c2 = -1 * np.sqrt((gamma/2)/2) * dark_state * excited_state.dag() #Excited to dark decay pathway
c_ops_list.append(c1)
c_ops_list.append(c2)

#Solve Linbladian for defined system
solution = qp.mesolve(three_level_hamiltonian, initial_dm, tlist = times,c_ops = c_ops_list, e_ops = [expectation_g, expectation_e, expectation_d])

system_evolution_dm_ground = solution.expect[0] #Ground state population evolution
system_evolution_dm_excited = solution.expect[1] #Excited state population evolution
system_evolution_dm_dark = solution.expect[2] #Dark state population evolution

#Plot resulting behaviour
plt.plot(times, system_evolution_dm_ground, label = 'Ground State')
plt.plot(times, system_evolution_dm_excited, label = 'Excited State')
plt.plot(times, system_evolution_dm_dark, label = 'Dark State')
plt.xlabel("Time (s)")
plt.ylabel("Occupation Probability")
plt.title("Three Level System Dynamics")
plt.legend(loc = 'upper right')
#plt.savefig("C:\\UniDurham\\GradCourseQLM\\AtomLightInteractions\\ProblemSheets\\3LevelSystemDynamics.png", transparent = None, dpi = 'figure')
plt.show()