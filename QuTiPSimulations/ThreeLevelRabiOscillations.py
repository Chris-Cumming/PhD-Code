# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:47:09 2024

@author: 13ccu
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qp

dim_system = 3 #Dimension of Hilbert space being considered

#Define hamiltonian and states of 3 level system, with one dark state

H3 = 2 * np.pi * np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]) #Same as two level but with extra zero terms

three_level_hamiltonian = qp.Qobj(H3)

ground_state = qp.fock(3, 0)
excited_state = qp.fock(3, 1)
dark_state = qp.fock(3, 2)

expectation_g = ground_state * ground_state.dag()
expectation_e = excited_state * excited_state.dag()
expectation_d = dark_state * dark_state.dag()

#Initial state of system is ground state - described by density matrix
initial_dm = qp.fock_dm(3, 0)

#Define time duration of system
times = np.linspace(0, 15, 1000)

#Define collapse operators of system
c_ops_list = []
gamma = 1
c1 = -1 * np.sqrt((gamma/2)/2) * ground_state * excited_state.dag() #Excited to ground
c2 = -1 * np.sqrt((gamma/2)/2) * dark_state * excited_state.dag() #Excited to dark
c_ops_list.append(c1)
c_ops_list.append(c2)

solution = qp.mesolve(three_level_hamiltonian, initial_dm, tlist = times,c_ops = c_ops_list, e_ops = [expectation_g, expectation_e, expectation_d])

system_evolution_dm_ground = solution.expect[0]
system_evolution_dm_excited = solution.expect[1]
system_evolution_dm_dark = solution.expect[2]

plt.plot(times, system_evolution_dm_ground, label = 'Ground State')
plt.plot(times, system_evolution_dm_excited, label = 'Excited State')
plt.plot(times, system_evolution_dm_dark, label = 'Dark State')
plt.xlabel("Time (s)")
plt.ylabel("Occupation Probability")
plt.title("Three Level System Dynamics")
plt.legend(loc = 'upper right')
#plt.savefig("C:\\UniDurham\\GradCourseQLM\\AtomLightInteractions\\ProblemSheets\\3LevelSystemDynamics.png", transparent = None, dpi = 'figure')
plt.show()