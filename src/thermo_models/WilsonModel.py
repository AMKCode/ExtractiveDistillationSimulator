import numpy as np
import os, sys
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 

import utils.AntoineEquation as AE
from thermo_models.VLEModelBaseClass import *


class WilsonModel(VLEModel):
    """
    A class representing a thermodynamic model based on Wilson.

    This model calculates the conversion between liquid mole fraction and vapor mole fraction
    based on Wilson's model.

    Args:
        num_comp (int): The number of components of the system -- typically 2 or 3
        P_sys (float): The system pressure in units compatible with the vapor pressures.
        Lambdas (dict): Dictionary keys are tuples (i,j) that indicate the lambda coefficient with corresponding value

    Methods:
        get_activity_coefficient: Using the known Lambda values, the gamma activity coefficients are computed according to Wilson's Equation
        get_vapor_pressure: Computes the vapor pressure for each component at a given temperature.

    Reference: MULTICOMPONENT EQUILIBRIA—THE WILSON EQUATION, R. V. Orye and J. M. Prausnitz.
    Industrial & Engineering Chemistry 1965 57 (5), 18-26.  DOI: 10.1021/ie50665a005
    """

    #CONSTRUCTOR 
    def __init__(self, num_comp:int, P_sys:float, Lambdas:dict, partial_pressure_eqs:AE):
        self.num_comp = num_comp
        self.P_sys = P_sys
        self.Lambdas = Lambdas
        self.partial_pressure_eqs = partial_pressure_eqs

    def get_activity_coefficient(self, x_):
        #From Equation 15 in Prausnitz, the gamma values are computed
        #This will work for any number of components

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
                print("x_", x_)
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
    



if __name__=='__main__':
    #Test a Ternary Mixture of Acetone, Methyl Acetate, Methanol
    # Acetone = 1
    # Methyl Acetate = 2
    # Methanol = 3
    # Lambda values from Prausnitz Table IV
    P_sys = 1.0325
    num_comp = 3
    x_ = [0.3, 0.2, 0.5]
    Lambdas = {
        (1,1) : 1.0,
        (1,2) : 0.5781,
        (1,3) : 0.6917,
        (2,1) : 1.3654,
        (2,2) : 1.0,
        (2,3) : 0.6370,
        (3,1) : 0.7681,
        (3,2) : 0.4871,
        (3,3) : 1.0
        }
    
    #Antoine parameters for Acetone
    Ace_A = 4.42448
    Ace_B = 1312.253
    Ace_C = -32.445

    #Antoine parameters for Methyl Acetate
    MA_A = 4.68206
    MA_B = 1642.54
    MA_C = -39.764

    #Antoine parameters for Methanol
    Me_A = 5.20409
    Me_B = 1581.341
    Me_C = -33.5

    #Antoine Equations 
    Acetate_antoine = AE.AntoineEquation(Ace_A, Ace_B, Ace_C)
    MethylAcetate_antoine = AE.AntoineEquation(MA_A, MA_B, MA_C)
    Methanol_antoine = AE.AntoineEquation(Me_A, Me_B, Me_C)

    Acetone_MethylAcetate_Methanol = WilsonModel(num_comp, P_sys, Lambdas,[Acetate_antoine, MethylAcetate_antoine, Methanol_antoine])  
    print(Acetone_MethylAcetate_Methanol.get_activity_coefficient(x_))
