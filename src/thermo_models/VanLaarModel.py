import numpy as np
from utils.AntoineEquation import *
from thermo_models.VLEModelBaseClass  import *

class VanLaarModel(VLEModel):
    """
    A class representing a thermodynamic model based on Van Laar.

    This model calculates the conversion between liquid mole fraction and vapor mole fraction
    based on Van Laar's model.

    Args:
        num_comp (int): The number of components in the system.
        P_sys (float): The system pressure in units compatible with the vapor pressures.
        comp_names (list): The names of the components in the system.
        partial_pressure_eqs (AntoineEquationBase10): The Antoine equations for each component.
        A_coeff (dict): A dictionary of Van Laar A coefficients for the components.
        use_jacobian (bool, optional): Flag to determine whether to use the Jacobian matrix in calculations.
 
    Attributes:
        P_sys (float): The system pressure used in the calculations.
        A_coeff (dict): A dictionary of Van Laar A coefficients for the components.
    """
    
    def __init__(self, num_comp: int, P_sys: float, comp_names, partial_pressure_eqs: AntoineEquationBase10, A_coeff: dict, use_jacobian=False):
        super().__init__(num_comp, P_sys, comp_names, partial_pressure_eqs, use_jacobian)
        self.A_coeff = A_coeff

    def get_activity_coefficient(self, x_array, Temp:float):
        #Assert that A_coeff[(i,i)] = 1
        for i in range(1, self.num_comp+1):
            if (self.A_coeff[(i,i)] != 0):
                raise ValueError('A Coefficients entered incorrectly')
            
        gammas = []
        z_array = [] # First compute the mole fractions 
        for i in range(1, self.num_comp+1):
            denom = 0
            for j in range(1, self.num_comp+1): #summation over j term in denominator
                if (i == j): # Aii = 0
                    denom += (x_array[j-1])  # "If Aji / Aij = 0 / 0, set Aji / Aij = 1" -- Knapp Thesis
                else:
                    denom += (x_array[j-1] * self.A_coeff[(j,i)] / self.A_coeff[(i,j)])
            z_array.append((x_array[i - 1] / denom))
        
        for k in range(1, self.num_comp+1):
            term1 = 0 # summation of Aki * zi
            term2 = 0 # summation of Aki * zk * zi
            term3 = 0 # The sum of sum of Aji * (Akj / Ajk) * zj * zi
            for i in range(1, self.num_comp+1):
                term1 += (self.A_coeff[(k,i)] * z_array[i-1])
                term2 += (self.A_coeff[(k,i)] * z_array[k-1] * z_array[i-1]) 
            for j in range(1, self.num_comp+1):
                for i in range(1, self.num_comp+1):
                    if (j == k or i == k): # "If Aji / Aij = 0 / 0, set Aji / Aij = 1" -- Knapp
                        pass
                    else:                      
                        term3 += (self.A_coeff[(j,i)] * self.A_coeff[(k,j)] / self.A_coeff[(j,k)] * z_array[j-1] * z_array[i-1])
            gammas.append(math.exp((term1 - term2 - term3)/Temp))
        return gammas

    def get_vapor_pressure(self, Temp)->np.ndarray:
        """
        Computes the vapor pressure for each component at a given temperature.
        
        Args:
            Temp (float): The temperature at which to compute the vapor pressure.
            
        Returns:
            np.ndarray: The vapor pressure for each component.
        """
        vap_pressure_array = []
        for partial_pressure_eq in self.partial_pressure_eqs:
            vap_pressure_array.append(partial_pressure_eq.get_partial_pressure(Temp))
        return np.array(vap_pressure_array)