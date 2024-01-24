import numpy as np
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from thermo_models.VLEModelBaseClass import *


class WilsonModel(VLEModel):
    """
    A class representing a thermodynamic model based on Wilson.

    This model calculates the conversion between liquid mole fraction and vapor mole fraction
    based on Wilson's model.

    Args:
        num_comp (int): The number of components of the system -- typically 2 or 3.
        P_sys (float): The system pressure in units compatible with the vapor pressures.
        Lambdas (dict): Dictionary keys are tuples (i,j) that indicate the lambda coefficient with corresponding value.
        comp_names (list): The names of the components in the system.
        partial_pressure_eqs (AntoineEquationBase10, optional): The Antoine equations for each component.
        use_jacob (bool, optional): Flag to determine whether to use the Jacobian matrix in calculations.

    Reference: 
        MULTICOMPONENT EQUILIBRIA—THE WILSON EQUATION, R. V. Orye and J. M. Prausnitz.
        Industrial & Engineering Chemistry 1965 57 (5), 18-26. DOI: 10.1021/ie50665a005
    """

    def __init__(self, num_comp: int, P_sys: float, comp_names, Lambdas: dict, partial_pressure_eqs=None, use_jacob=False):
        super().__init__(num_comp, P_sys, comp_names, partial_pressure_eqs, use_jacob)
        self.Lambdas = Lambdas

    def get_activity_coefficient(self, x_, Temp = None):
        #Assert that Lambdas[(i,i)] = 1
        for i in range(1, self.num_comp+1):
            if (self.Lambdas[(i,i)] != 1):
                raise ValueError('Lambda Coefficients entered incorrectly')
            
        gamma_list = []
        for k in range(1, self.num_comp+1):
            gamma_k = 1
            log_arg = 0
            for j in range(1, self.num_comp+1):
               log_arg += ( x_[j-1] * self.Lambdas[(k,j)] )
            if log_arg <= 0:
                raise ValueError
            gamma_k -= np.log(log_arg)

            for i in range(1, self.num_comp+1):
                dividend = (x_[i-1] * self.Lambdas[(i,k)] )
                divisor = 0
                for j in range(1, self.num_comp+1):
                    divisor += (x_[j-1] * self.Lambdas[(i,j)] )
                gamma_k -= (dividend / divisor)
            gamma_list.append(np.exp(gamma_k))
        return np.array(gamma_list)
    
    def get_vapor_pressure(self, Temp)->np.ndarray:
        """
        Args:
            Temp (float): The temperature at which to compute the vapor pressure.      
        Returns:
            np.ndarray: The vapor pressure for each component.
        """
        vap_pressure_array = []
        for partial_pressure_eq in self.partial_pressure_eqs:
            vap_pressure_array.append(partial_pressure_eq.get_partial_pressure(Temp))
        return np.array(vap_pressure_array)
