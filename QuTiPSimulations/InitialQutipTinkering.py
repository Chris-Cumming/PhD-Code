# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:31:50 2024

@author: xxvh83
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *


initial_state_ket = Qobj([[1], [2], [3], [4], [5]])
print(initial_state_ket)
initial_state_bra = initial_state_ket.trans()
print(initial_state_bra)


initial_array = np.array([2, 4, 6, 8, 10])
new_state_ket = Qobj(initial_array)
print(new_state_ket)
new_state_bra = new_state_ket.trans()
print(new_state_bra)


#Basis is used to create state function ket of dimension 3 with excitation in level 3
print(basis(3, 2)) #Could also use fock
#Empty ket
print(zero_ket(5))
#Density matrix of basis state, of dimension 4 and with excitation in level 2
print(fock_dm(4, 2))

#Used for harmonic oscillator systems where the creation and annihilation (ladder) operators are used 
print(destroy(2))
print("Creation Operator for SHO")
print(create(2))

#Can also use sigmap and sigmam for the raising and lowering operators of spin 1/2 systems
print(sigmam())
print("Raising Operator for Spin 1/2 System")
print(sigmap())

#Higher spin system raising. and lowering operators can be computed from jmat function

#The .unit attribute will normalise a state to unity - useful when creating superposition of states
ket = (basis(5, 0) + basis(5, 1)).unit()
print(ket)

