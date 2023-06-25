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
    def __init__(self, num_comp: int, P_sys: float, partial_pressure_eqs: AntoineEquation, A, B):
        super().__init__(num_comp, P_sys)
        self.partial_pressure_eqs = partial_pressure_eqs
        self.A = A
        self.B = B

    def get_activity_coefficient(self, x_array):

        gamma1 = np.exp(self.A/(np.power(1+((self.A*x_array[0])/(self.B*x_array[1])), 2)))
        gamma2 = np.exp(self.B/(np.power(1+((self.B*x_array[1])/(self.A*x_array[0])), 2)))
        
        return np.array([gamma1, gamma2])

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
            
            