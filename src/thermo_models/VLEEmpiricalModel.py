import numpy as np
import os, sys
from typing import Callable
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from thermo_models.VLEModelBaseClass import *

class VLEEmpiricalModel(VLEModel):
    def __init__(self, func_xtoy: Callable[[np.ndarray], np.ndarray], 
                 func_ytox: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Initializes the VLEEmpiricalModel with conversion functions.

        Args:
            func_xtoy (Callable): Function to convert from x to y.
            func_ytox (Callable): Function to convert from y to x.
        """
        self.func_xtoy = func_xtoy
        self.func_ytox = func_ytox
        #tests if the conversion back and forth is valid

    def convert_x_to_y(self, x_array: np.ndarray) -> np.ndarray:
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (np.ndarray): Liquid mole fraction of each component.

        Returns:
            solution (np.ndarray): The solution by using the equation provided in self.func_xtoy during initialization.

        Raises:
            ValueError: If the sum of mole fractions is not equal to 1.
        """
        if np.sum(x_array) != 1:
            raise ValueError("The sum of mole fraction must be equal to 1")

        solution = self.func_xtoy(x_array)

        if np.sum(solution) != 1:
            raise ValueError("The sum of mole fraction must be equal to 1")

        return solution
    
    def convert_y_to_x(self, y_array: np.ndarray) -> np.ndarray:
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (np.ndarray): Liquid mole fraction of each component.

        Returns:
            solution (np.ndarray): The solution by using the equation provided in self.func_xtoy during initialization.

        Raises:
            ValueError: If the sum of mole fractions is not equal to 1.
        """
        if np.sum(y_array) != 1:
            raise ValueError("The sum of mole fraction must be equal to 1")

        solution = self.func_ytox(y_array)

        if np.sum(solution) != 1:
            raise ValueError("The sum of mole fraction must be equal to 1")

        return solution
    
    def compute_Txy(self, vars: np.ndarray, x_array: np.ndarray) -> list:
        raise NotImplementedError("This method should not be called for empirical models")

    def compute_Txy2(self, vars: np.ndarray, y_array: np.ndarray) -> list:
        raise NotImplementedError("This method should not be called for empirical models")