# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:04:21 2024

@author: xxvh83
"""

import numpy as np
import qutip as qp
import matplotlib.pyplot as plt



#Solving rabi oscillation system

dim_system = 2 #Dimension of Hilbert space of system being considered, 2 level system
occupied_state = 0 #Indicates which level of the system is initially occupied, 0 refers to ground state, 1 to first excited state, etc
unoccupied_state = 1 #Indicates the excited state level which is initially unoccupied at t = 0
rabi_freq = 4
detuning_freq = 0

#Define hamiltonian of system
H_rabi = 2 * np.pi * 0.5 * qp.sigmax() * rabi_freq
H_det = - 2 * np.pi * 0.5 * qp.sigmaz() * detuning_freq

H_total = H_rabi + H_det

initial_state = qp.fock(dim_system, occupied_state) #System begins in initial state
g = initial_state #ground state is initial state
e = qp.fock(dim_system, unoccupied_state) #excited state
times = np.linspace(0, 4 * 1/rabi_freq, 1000) #Define times for system to be solved for
result_size = np.size(times)

#Solve system dynamics
result = qp.sesolve(H_total, initial_state, times)

#Initialise arrays to store values for probability of state being occupied
ground_state_population = np.zeros(result_size)
excited_state_population = np.zeros(result_size)

system_evolution = result.states
print(system_evolution)


for i in range(result_size):
    state_evolution = system_evolution[i].full()
    ground_state_population[i] = abs(state_evolution[0])**2
    excited_state_population[i] = abs(state_evolution[1])**2

plt.xlabel("Time (s)")
plt.ylabel("Probability of Level Occupation")
plt.plot(times, ground_state_population, label = 'Ground State')
plt.plot(times, excited_state_population, label = 'Excited State')
plt.legend(loc = 'upper right')
plt.title("sesolver")
plt.show()




#Attempt to solve with mesolve instead with no collapose operators - Master equation is then equivalent to Liouville equation
#Identical to TDSE which is what sesolve uses

initial_dm = qp.fock_dm(2, 0)
times = np.linspace(0, 4 * 1/rabi_freq, 1000)

expectation_g = g * g.dag()
#print(expectation_g)
expectation_e = e * e.dag()
#print(expectation_e)

solution = qp.mesolve(H_total, initial_dm, tlist = times, e_ops = [expectation_g, expectation_e])

system_evolution_dm_ground = solution.expect[0]
system_evolution_dm_excited = solution.expect[1]

plt.plot(times, system_evolution_dm_ground, label = 'Ground State')
plt.plot(times, system_evolution_dm_excited, label = 'Excited State')
plt.title("mesolver")
plt.show()

#Now use mesolve to account for spontaneous emission in two level system

#Create collapse operators that describe the process of spontaneous emission between ground and excited state
c_ops_list = []
gamma = 1
c1 = -1 * np.sqrt(gamma) * g * e.dag()
c_ops_list.append(c1)

solution = qp.mesolve(H_total, initial_dm, tlist = times,c_ops = c_ops_list, e_ops = [expectation_g, expectation_e])

system_evolution_dm_ground = solution.expect[0]
system_evolution_dm_excited = solution.expect[1]

plt.plot(times, system_evolution_dm_ground, label = 'Ground State')
plt.plot(times, system_evolution_dm_excited, label = 'Excited State')
plt.title("mesolver")
plt.show()



