# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:30:51 2024

@author: xxvh83
"""

import numpy as np
import qutip as qp

bloch = qp.Bloch() #Initialises the Bloch class
bloch.make_sphere() #Creates sphere
bloch.show() #Command required to display sphere

#Add a single data point to Bloch sphere

point = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
bloch.add_points(point) #Add the point to the bloch sphere
bloch.render() #Not strictly necessary
bloch.show()

#Add a vector to the sphere

bloch.clear() #Clear the current bloch sphere
vector = [0, 1, 0] #Define vector in 3D space we want to display
bloch.add_vectors(vector) #Add the vector to the bloch sphere
bloch.show()

#Add a basis state to the sphere
up = qp.basis(2, 0) #Define basis state, equivalent to 0 ket
bloch.add_states(up) #Add quantum object state - crucially not the same as the 3D vector representation
bloch.show()

#Now plot the 3 states associated with the cartesian coordinates
bloch.clear()
x = (qp.basis(2, 0) + (1 + 0j) * qp.basis(2, 1)).unit()
y = (qp.basis(2, 0) + (0 + 1j) * qp.basis(2, 1)).unit()
z = (qp.basis(2, 0) + (0 + 0j) * qp.basis(2, 1)).unit()
bloch.add_states([x, y, z])
bloch.show()

#Now plot the same vectors but in explicit vector notation instead of quantum basis states
bloch.clear()
x = [1, 0, 0]
y = [0, 1, 0]
z = [0, 0, 1]
vectors = [x, y, z]
bloch.add_vectors(vectors)
bloch.show()

#Note how up to this point each line added has been a different colour - this is not necessarily true when you parse the Bloch class a dataset
#Here we parse one data set consisting of 20 data points - these are then all displayed in the same colour as Bloch class assumes
#that these are all the same data point just at different times.
bloch.clear()
theta = np.linspace(0, 2 * np.pi, 20)
xp = np.cos(theta)
yp = np.sin(theta)
zp = np.zeros(20)
points = [xp, yp, zp]
bloch.add_points(points)
bloch.show()

#Here we add an extra data set which will now be displayed in a different colour to the previous one
xz = np.zeros(20)
yz = np.sin(theta)
zz = np.cos(theta) 
new_points = [xz, yz, zz]
bloch.add_points(new_points)
bloch.show()

#It is possible to further customise the sphere - transparancy, size, labels, fonts, etc - see the API documentation on Bloch class.
#Can also gain this information by directly printing the bloch object

print(bloch)

#Perhaps most usefully we can create animations with the Bloch sphere
#This can be done either by using Matplotlib to directly animate it or alternatively we can save an image of the bloch sphere 
#at each time and then animate it.
#Both methods have examples given in the user guide under the bloch sphere section








