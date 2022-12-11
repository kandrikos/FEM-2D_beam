import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import itertools
import seaborn as sns

from finite_elements_analysis import FiniteElementsAnalysis
import karhunen_loeve_solver

import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulation(FiniteElementsAnalysis):

    def __init__(self, lenX, lenY, n_elemX, n_elemY, poisson, thickness, n_simulations):
        super().__init__(lenX, lenY, n_elemX, n_elemY, poisson, thickness)
        self.n_simulations = n_simulations  # number of simulations

    def karhunen_loeve(self):
        return karhunen_loeve_solver.run_KL(M=self.n_simulations)

    def get_load(self):
        return np.random.normal(10, 2, self.n_simulations)

    def run_MonteCarlo(self):
        right_bottom_node_dispY = []
        E_kl = self.karhunen_loeve()
        loads = self.get_load()
        print("\nRunning Finite Element Analysis for each realization ...")
        for r in tqdm(range(len(E_kl))):
            realization = E_kl[r]

            el = 0
            self.global_stiffness = np.zeros((self.g_dofs, self.g_dofs))
            for el, i in enumerate(itertools.cycle(range(len(realization)))):
                
                k_element_global = self.get_element_global_stiffness(element_dof=self.elem_dofs[el], E=realization[i])
                self.global_stiffness += k_element_global
                if el == self.elem_dofs.shape[0] - 1:
                    break
            
            self.set_boundary_conditions()
            self.set_force_vector(load=loads[r])
            node_displacement = self.get_displacements(flag=True)
            right_bottom_node_dispY.append(node_displacement)
        sns.distplot(right_bottom_node_dispY, bins=10)
        plt.xlabel("Displacement")
        plt.ylabel("Density")
        plt.savefig(os.path.join(os.getcwd(), f"PDF_{self.n_simulations}_simulations.png"))
        with open(f"Displacements_{self.n_simulations}_simulations.txt", 'w') as f: # save the displacements of the right bottom corner to a .txt
            for item in right_bottom_node_dispY:
                f.write("%s\n" % item)

if __name__ == '__main__':
    mc = MonteCarloSimulation(n_elemX=40, n_elemY=10, lenX=4, lenY=1, poisson=0.3, thickness=0.2, n_simulations=100)
      
    mc.rectangular_mesh()
    mc.get_dofs()
    mc.run_MonteCarlo()