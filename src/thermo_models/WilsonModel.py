from ThermodynamicModel import ThermodynamicModel
import numpy as np

class WilsonModel(ThermodynamicModel):
    """
    A class representing a thermodynamic model based on Wilson.

    This model calculates the conversion between liquid mole fraction and vapor mole fraction
    based on Wilson's model.

    Args:
        num_comp (int): The number of components of the system -- typically 2 or 3
        P_sys (float): The system pressure in units compatible with the vapor pressures.
        Lambdas (dict): Dictionary keys are tuples (i,j) that indicate the lambda coefficient with corresponding value

    Methods:
        get_activity: Using the known Lambda values, the gamma activity coefficients are computed according to Wilson's Equation

        
    Reference: MULTICOMPONENT EQUILIBRIA—THE WILSON EQUATION, R. V. Orye and J. M. Prausnitz.
    Industrial & Engineering Chemistry 1965 57 (5), 18-26.  DOI: 10.1021/ie50665a005
    """

    #CONSTRUCTOR 
    def __init__(self, num_comp:int, P_sys:float, Lambdas:dict):
        self.num_comp = num_comp
        self.P_sys = P_sys
        self.Lambdas = Lambdas
        #Assert that Lambdas[(i,i)] = 1
        for i in range(num_comp):
            if (Lambdas[(i,i)] != 1):
                raise ValueError('Lambda Coefficients entered incorrectly')


    #From Equation 15 in Prausnitz, the gamma values are computed
    #This will work for any number of components
    def get_activity(self, num_comp, Lambdas, x_)
        gamma_list = []
        for k in range(num_comp):
            gamma_k = 1
            log_arg = 0
            for j in range(num_comp):
               log_arg += ( x_[j] * Lambdas[(k,j)] )
            gamma_k -= np.log(log_arg)

            for i in range(num_comp):
                dividend = (x_[i] * Lambdas[(i,k)] )
                divisor = 0
                for j in range(num_comp):
                    divisor += (x_[j] * Lambdas[(i,j)] )
                gamma_k -= (dividend / divisor)
            gamma_list.append(gamma_k)



    

        