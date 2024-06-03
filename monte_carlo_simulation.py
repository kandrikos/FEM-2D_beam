import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import itertools
import seaborn as sns

from finite_elements_analysis import FiniteElementsAnalysis
import karhunen_loeve_solver
import arguments

import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulation(FiniteElementsAnalysis):

    def __init__(self, lenX, lenY, n_elemX, n_elemY, poisson, thickness, n_simulations):
        super().__init__(lenX, lenY, n_elemX, n_elemY, poisson, thickness)
        self.n_simulations = n_simulations  # number of simulations

    def karhunen_loeve(self):
        return karhunen_loeve_solver.run_KL(M=self.n_simulations)

    def get_load(self):
        return np.random.normal(arguments.determ_load, 2, self.n_simulations)

    def run_MonteCarlo(self):
        right_bottom_node_dispY = []
        E_kl = self.karhunen_loeve()
        loads = self.get_load()
        self.rectangular_mesh()
        self.get_dofs()
        for r in tqdm(range(len(E_kl)), desc="Running Finite Element Analysis for each realization"):
            realization = E_kl[r]
            self.global_stiffness = np.zeros((self.g_dofs, self.g_dofs))
            for el, i in enumerate(itertools.cycle(range(len(realization)))):
                k_element_global = self.get_element_global_stiffness(element_dof=self.elem_dofs[el], E=realization[i])
                self.global_stiffness += k_element_global
                if el == self.elem_dofs.shape[0] - 1:
                    break
            self.set_boundary_conditions()
            self.set_force_vector(load=loads[r])
            node_displacement = self.get_displacements(return_displ=True)
            right_bottom_node_dispY.append(node_displacement)
        sns.distplot(right_bottom_node_dispY, bins=10)
        plt.xlabel("Displacement")
        plt.ylabel("Density")
        plt.savefig(os.path.join(os.getcwd(), f"PDF_{self.n_simulations}_simulations.png"))

if __name__ == '__main__':
    mc = MonteCarloSimulation(n_simulations=arguments.n_simulations, **arguments.shared_args)
    mc.run_MonteCarlo()
