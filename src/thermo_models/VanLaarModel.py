import numpy as np
from utils.AntoineEquation import *
from thermo_models.VLEModelBaseClass  import *

class VanLaarModel(VLEModel):
    """
    A class representing a thermodynamic model based on Van Laar.

    This model calculates the conversion between liquid mole fraction and vapor mole fraction
    based on Van Laar's model.

    Args:
        P_sys (float): The system pressure in units compatible with the vapor pressures.
 
    Attributes:
        P_sys (float): The system pressure used in the calculations.

    Methods:
        convert_x_to_y(x, psat, A, B):
            Convert liquid mole fraction to vapor mole fraction based on Van Laar.
        convert_y_to_x(y, psat, A, B):
            Convert vapor mole fraction to liquid mole fraction based on Van Laar.
    """
    def __init__(self, num_comp: int, P_sys: float, A_coeff:dict, comp_names, partial_pressure_eqs: AntoineEquationBase10, A = None, B= None):
        super().__init__(num_comp, P_sys, comp_names)
        self.A_coeff = A_coeff
        self.partial_pressure_eqs = partial_pressure_eqs
        self.A = A
        self.B = B
        self.use_jacobian = False

    def get_activity_coefficient(self, x_array, Temp:float):
        if (self.num_comp == 2):
            gamma1 = np.exp(self.A/(np.power(1+((self.A*x_array[0])/(self.B*x_array[1])), 2)))
            gamma2 = np.exp(self.B/(np.power(1+((self.B*x_array[1])/(self.A*x_array[0])), 2)))
            
            return np.array([gamma1, gamma2])
        else:
            #Assert that A_coeff[(i,i)] = 1
            for i in range(1, self.num_comp+1):
                if (self.A_coeff[(i,i)] != 0):
                    raise ValueError('Lambda Coefficients entered incorrectly')
                
            gammas = []
            z_array = [] # First compute the mole fractions 
            for i in range(1, self.num_comp+1):
                denom = 0
                for j in range(1, self.num_comp+1): #summation over j term in denominator
                    if (i == j): # Aii = 0
                        denom += (x_array[j-1])  # "If Aji / Aij = 0 / 0, set Aji / Aij = 1" -- Knapp
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
                            # term3 += (self.A_coeff[(j,i)] * z_array[j-1] * z_array[i-1])
                        else:
                            # if self.A_coeff[i, j] == 0 and self.A_coeff[i, j] == 0:
                            #     term3 += (self.A_coeff[(j,i)]  * z_array[j-1] * z_array[i-1])
                            # else:                        
                            term3 += (self.A_coeff[(j,i)] * self.A_coeff[(k,j)] / self.A_coeff[(j,k)] * z_array[j-1] * z_array[i-1])
                gammas.append(math.exp((term1 - term2 - term3)/Temp))
            # print(gammas)
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
            
            