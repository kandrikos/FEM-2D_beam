#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kostas Andrikos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class FiniteElementsAnalysis:

    def __init__(self, lenX, lenY, n_elemX, n_elemY, poisson, thickness):
        self.lenX = lenX
        self.lenY = lenY
        self.n_elemX = n_elemX
        self.n_elemY = n_elemY
        self.poisson = poisson
        self.thickness = thickness
        self.element_nodes = []
        self.elem_dofs = []
        self.g_dofs = 2 * (n_elemX + 1) * (n_elemY + 1) # global degrees of freedom

    def rectangular_mesh(self):
        """  
        Method for discretizing the domain using quadrilateral elements. Returns a 2d list of shape (num_of_elements, 4)
        (from book 'MATLAB Codes for Finite Element Analysis' by Ferreira, António J. M., Fantuzzi, Nicholas)
        """
        j = 1
        i = 1
        i1 = 0
        for j in range(self.n_elemY):
            for i in range(self.n_elemX):
                if i == 0 and j == 0:
                    i1 = 1
                else:
                    i1 += 1
                i2 = i1 + 1
                i4 = i1 + self.n_elemX + 1
                i3 = i2 + self.n_elemX + 1
                self.element_nodes.append([i1, i2, i3, i4])
            i1 += 1
            i2 += 1
        self.element_nodes = np.array(self.element_nodes)

    def get_dofs(self):
        '''
        Returns a list with the dofs of an element
        '''
        for j in range(len(self.element_nodes)):
            dofs = []
            for i in self.element_nodes[j]:
                dofs.append(2*i - 1)
                dofs.append(2*i)
            self.elem_dofs.append(dofs)
        self.elem_dofs = np.array(self.elem_dofs) - 1

    def local_stiffness_matrix(self, E):
        """
        Calculates the stiffness matrix of a quadrilateral element
        """
        a = self.lenX / (2 * self.n_elemX)
        b = self.lenY / (2 * self.n_elemY)
        r = a/b
        rho = (1 - self.poisson)/2
        mu = (1 + self.poisson) * 3/2
        lamda = (1 - 3*self.poisson)/2

        k = np.zeros((8, 8))
        k[0, 0] = k[2, 2] = k[4, 4] = k[6, 6] = 4/r + 4*rho*r
        k[1, 1] = k[3, 3] = k[5, 5] = k[7, 7] = 4*r + 4*rho/r
        k[2, 0] = k[0, 2] = k[6, 4] = k[4, 6] = -4/r + 2*rho*r
        k[4, 0] = k[0, 4] = k[6, 2] = k[2, 6] = -2/r - 2*rho*r
        k[6, 0] = k[0, 6] = k[4, 2] = k[2, 4] = 2/r - 4*rho*r
        k[3, 1] = k[1, 3] = k[7, 5] = k[5, 7] = 2*r - 4*rho/r
        k[5, 1] = k[1, 5] = k[7, 3] = k[3, 7] = -2*r - 2*rho/r
        k[7, 1] = k[1, 7] = k[5, 3] = k[3, 5] = -4*r + 2*rho/r
        k[1, 0] = k[0, 1] = k[7, 2] = k[2, 7] = k[6,3] = k[3, 6] = k[5, 4] = k[4, 5] = mu
        k[3, 0] = k[0, 3] = k[6, 1] = k[1, 6] = k[5,2] = k[2, 5] = k[7, 4] = k[4, 7] = -lamda
        k[5, 0] = k[0, 5] = k[4, 1] = k[1, 4] = k[3,2] = k[2, 3] = k[7, 6] = k[6, 7] = -mu
        k[7, 0] = k[0, 7] = k[1, 2] = k[2, 1] = k[4,3] = k[3, 4] = k[6, 5] = k[5, 6] = lamda

        return E * self.thickness / (12 * (1 - self.poisson**2)) * k

    def get_element_global_stiffness(self, element_dof, E):
        """
        For a given element (defined by its dofs) the method returns its global stiffness matrix
        """
        k_element_local = self.local_stiffness_matrix(E)
        self.k_element_glob = np.zeros((self.g_dofs, self.g_dofs))
        c1 = 0
        for i in element_dof:
            c2 = 0
            for j in element_dof:
                self.k_element_glob[i, j] = k_element_local[c1, c2]
                c2 += 1
            c1 += 1
        return self.k_element_glob

    def get_global_stiffness(self, young_modulus):
        """
        Returns the global stiffness matrix of the system. But first the global stiffness matrix of each element must be calculated
        """
        self.global_stiffness = np.zeros((self.g_dofs, self.g_dofs))
        for el in range(len(self.elem_dofs)):
            element_global_stiffness = self.get_element_global_stiffness(element_dof=self.elem_dofs[el], E=young_modulus)
            self.global_stiffness += element_global_stiffness

    def set_boundary_conditions(self):
        """
        Returns a list which contains the dofs of the left edge of the beam.
        If we want to set different boundary conditions, we should adjust the content of this list to include the dofs we want to constrain.
        """
        self.constrained_dofs = np.unique(self.elem_dofs[::self.n_elemX][:, [0, 1, -2, -1]]) # constrains the left edge
        # self.constrained_dofs = np.unique(self.elem_dofs[self.n_elemX-1::self.n_elemX][:, 2:6]) # constrains the right edge
        # self.constrained_dofs = np.array(sorted(np.concatenate
        # (
        #     (np.unique(self.elem_dofs[::self.n_elemX][:, [0, 1, -2, -1]]),
        #     np.unique(self.elem_dofs[self.n_elemX-1::self.n_elemX][:, 2:6]))))
        # )

    def set_force_vector(self, load):
        """
        Returns the forces field. In this example the load is applied to the vertical direction (negative-y) of the upper right node (last dof of the system).
        If we want to change the loading condition we should adjust the content of this list to include the dofs we want to carry loads.
        """
        self.forces = np.zeros(self.g_dofs) # Define the external force vector
        self.forces[-1] = -load # the load is applied to the last dof - direction to negative y
        # self.forces[-2] = -5 * load
        # self.forces[-int((2*self.n_elemX + 1) / 2) + 1] = -load  # the load is applied to the middle-point of the upper surface - direction to negative y

    def get_displacements(self, return_displ=False):
        """
        Returns the displacement field
        """
        for i in self.constrained_dofs:               
            self.global_stiffness[:, i] = 0           # For each constrained dof i, set the elements of i-row & i-column
            self.global_stiffness[i, :] = 0           # of the global stiffness matrix to zero
            self.global_stiffness[i, i] = 1e10        # Assign a very large positive number to the i-diagonal element

        # Compute the displacement vector
        self.displacements = np.dot(np.linalg.inv(self.global_stiffness), self.forces)  # U = K^-1 * P
        self.displacements = self.displacements.reshape(self.n_elemY + 1, self.n_elemX + 1, 2)
        if return_displ:
            return self.displacements[0][-1][1]
        else:
            print(f"\nDisplacement of the bottom right corner of the beam (m):\n{self.displacements[0][-1][1]}")
            
    def mesh_plot(self):
        nx, ny = (self.n_elemX + 1, self.n_elemY + 1)

        x = np.linspace(0, self.lenX, nx)
        y = np.linspace(0, self.lenY, ny)
        xv, yv = np.meshgrid(x, y)
        plt.scatter(xv, yv, marker=".")

        segs1 = np.stack((xv, yv), axis=2)
        segs2 = segs1.transpose(1,0,2)
        segs3 = segs1 + self.displacements
        segs4 = segs2 + self.displacements.transpose(1,0,2)
        
        plt.figure(figsize=(9, 3))
        plt.xlim(xmin=-0.1, xmax=self.lenX+0.1)
        plt.ylim(ymin=-0.15, ymax=self.lenY+0.1)

        plt.gca().add_collection(LineCollection(segs1, colors='C0'))
        plt.gca().add_collection(LineCollection(segs2, colors='C0'))
        plt.gca().add_collection(LineCollection(segs3, colors='C3'))
        plt.gca().add_collection(LineCollection(segs4, colors='C3'))
        plt.title("Initial vs. Deformed structure")
        plt.savefig("initial_vs_deformed_structure.png")
        plt.close()


if __name__ == '__main__':
    import arguments
    
    fea = FiniteElementsAnalysis(**arguments.shared_args)
    fea.rectangular_mesh()
    fea.get_dofs()
    fea.get_global_stiffness(young_modulus=arguments.determ_modulus)
    fea.set_boundary_conditions()
    fea.set_force_vector(load=arguments.determ_load)
    fea.get_displacements()
    fea.mesh_plot()
