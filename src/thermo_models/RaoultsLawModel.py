import numpy as np
from ThermodynamicModel import ThermodynamicModel

class RaoultsLawModel(ThermodynamicModel):
    """
    A class representing a thermodynamic model based on Raoult's law.

    This model calculates the conversion between liquid mole fraction and vapor mole fraction
    based on Raoult's law, which assumes ideal behavior for the components in the mixture.

    Args:
        P_sys (float): The system pressure in units compatible with the vapor pressures.

    Attributes:
        P_sys (float): The system pressure used in the calculations.

    Methods:
        convert_x_to_y(x_array, P_sat_array): 
            Convert liquid mole fraction to vapor mole fraction based on Raoult's law.
        convert_y_to_x(y_array, P_sat_array):
            Convert vapor mole fraction to liquid mole fraction based on Raoult's law.
    """
    
    def __init__(self, P_sys, antoine_eqs=None):
        self.P_sys = P_sys
        self.antoine_eqs = antoine_eqs
        
    def compute_composition(self, T_space): #do not know how to extend to multiple components
        P1 = self.antoine_eqs[0].get_partial_pressure(T_space)
        P2 = self.antoine_eqs[1].get_partial_pressure(T_space)
        x_1 = (self.P_sys-P2)/(P1-P2)
        y_1 = x_1*P1/self.P_sys
        return x_1, y_1
    
    
    def convert_x_to_y(self, x_array: np.ndarray, P_sat_array: np.ndarray):
        """
        Converts liquid mole fraction (x) to vapor mole fraction (y) based on Raoult's law.

        Args:
            x_array (np.ndarray): An array of liquid mole fractions for each component in the mixture.
            P_sat_array (np.ndarray): An array of saturation pressures for each component in the mixture.

        Returns:
            np.ndarray: An array of vapor mole fractions for each component in the mixture.
        """
        y_array = x_array * P_sat_array / self.P_sys
        return y_array
    
    def convert_y_to_x(self, y_array: np.ndarray, P_sat_array: np.ndarray):
        """
        Converts vapor mole fraction (y) to liquid mole fraction (x) based on Raoult's law.

        Args:
            y_array (np.ndarray): An array of vapor mole fractions for each component in the mixture.
            P_sat_array (np.ndarray): An array of saturation pressures for each component in the mixture.

        Returns:
            np.ndarray: An array of liquid mole fractions for each component in the mixture.
        """
        x_array = y_array * self.P_sys / P_sat_array
        return x_array




        
        