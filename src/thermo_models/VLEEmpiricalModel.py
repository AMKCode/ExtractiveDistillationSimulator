import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from typing import Callable
import random as rand

PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 

from thermo_models.VLEModelBaseClass  import *

class VLEEmpiricalModelBinary(VLEModel):
    def __init__(self, comp_names, func_xtoy: Callable[[float], float]) -> None:
        """Initialize the VLE empirical model.

        Args:
            num_comp (int): The number of components in the system.
            P_sys (float): The system pressure in units compatible with the vapor pressures.
            comp_names (list): The names of the components in the system.
            partial_pressure_eqs (Any): The equations or data for calculating partial pressures.
            func_xtoy (Callable): Function to convert mole fraction x to y.
            use_jacobian (bool, optional): Whether to use the Jacobian matrix in calculations.
        """
        partial_pressure_eqs = None
        super().__init__(2, None, comp_names, None, False)
        self.func_xtoy = func_xtoy


    def convert_x_to_y(self, x_array: np.ndarray):
        """Converts x to y using the provided function.

        Args:
            x_array (float): Mole fraction in the liquid phase.

        Returns:
            solution (float): Mole fraction in the vapor phase.
            message (str): Informational message.
        """
        x = x_array[0]
        solution = self.func_xtoy(x_array)
        return solution, "Does not use a solver"
    
    def convert_y_to_x(self, y_array: np.ndarray, x_guess: float = None):
        """Converts y to x by solving the provided function.

        Args:
            y (float): Mole fraction in the vapor phase.
            x_guess (float, optional): Initial guess for x. Defaults to a random value between 0 and 1.

        Returns:
            solution (float): Mole fraction in the liquid phase.
            message (str): Informational message.

        Raises:
            ValueError: If fsolve does not find a solution or solution is not between 0 and 1.
        """
        # Define a function that needs to be solved
        
        y = y_array[0]
        def func(x):
            return self.func_xtoy(x) - y

        # Initial guess for the liquid mole fractions
        if x_guess is None:
            x_guess = rand.uniform(0,1)
            
        # Iterate until solution is found or max iterations are reached
        for _ in range(500):  # limit the number of iterations
            solution, infodict, ier, mesg = fsolve(func, x_guess, full_output=True, xtol=1e-12)
            if ier == 1:  # fsolve succeeded
                # Check if the solution is valid (i.e., the sum of mole fractions is 1)
                if solution > 1 or solution < 0:
                    raise ValueError("Mole fractions must be between 0 and 1")
                return solution, mesg
            # fsolve failed, generate a new guess
            x_guess = rand.uniform(0,1)

        raise ValueError("fsolve failed to find a solution")
    
    def plot_binary_yx(self, data_points: int = 100, comp_id:int = 0) -> None:
        """Plot the Vapor-Liquid Equilibrium (VLE) y-x diagram for a binary mixture.

        Args:
            data_points (int, optional): Number of data points to generate for the plot. Defaults to 100.
            comp_id (int, optional): Component id (0 or 1) for which the diagram should be plotted. Defaults to 0.

        Raises:
            ValueError: If comp_id is not 0 or 1.
        """
        if comp_id not in [0, 1]:
            raise ValueError("comp_id must be either 0 or 1")

        x_space = np.linspace(0,1,data_points)
        y_space = [self.convert_x_to_y(x)[0] for x in x_space]

        # Adjust for comp_id = 1
        if comp_id == 1:
            x_space = 1 - x_space
            y_space = 1 - np.array(y_space)

        plt.figure(figsize=(8,6))
        plt.plot(x_space, y_space, label="y vs x")
        plt.xlabel(f"Component {comp_id} mole fraction in liquid phase, x")
        plt.ylabel(f"Component {comp_id} mole fraction in vapor phase, y")
        plt.title("Vapor-Liquid Equilibrium (VLE) y-x Diagram")
        plt.grid(True)
        plt.legend()
        plt.show()
    

    
