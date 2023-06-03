from .ThermodynamicModel import ThermodynamicModel
from ..utils.AntoineEquation import AntoineEquation

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
    def __init__(self, P_sys):
        self.P_sys = P_sys
    
    def convert_x_to_y(self, x_array, P_sat_array):
        y_array = []
        for index,x in enumerate(x_array):
            y_array.append(x*P_sat_array[index]/self.P_sys)
        return y_array
    
    def convert_y_to_x(self, y_array, P_sat_array):
        x_array = []
        for index,y in enumerate(y_array):
            x_array.append(y*self.P_sys/P_sat_array[index])
        return x_array