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

from utils.AntoineEquation import *
from thermo_models.VLEModelBaseClass  import *

class RaoultsLawModel(VLEModel):
    """
    This class represents a model of a system following Raoult's Law for
    a vapor-liquid equilibrium with ideal gas and ideal liquid phase.
    It inherits from the VLEModel base class and implements the specific 
    methods to compute gas partial fugacity, vapor pressure, and activity coefficient.
    
    Attributes:
        num_comp (int): The number of components in the system.
        P_sys (float): The total pressure of the system.
        partial_pressure_eqs (AntoineEquation): The Antoine equations for each component.
    """
    
    def __init__(self, num_comp: int, P_sys: float, partial_pressure_eqs: AntoineEquationBase10):
        """
        Initializes the RaoultsLawModel with the number of components, system pressure, 
        and Antoine equations for each component.
        
        Args:
            num_comp (int): The number of components in the system.
            P_sys (float): The total pressure of the system.
            partial_pressure_eqs (AntoineEquation): The Antoine equations for each component.
        """
        super().__init__(num_comp, P_sys)
        self.partial_pressure_eqs = partial_pressure_eqs
        
    def compute_gas_partial_fugacity(self,y_i:np.ndarray) -> np.ndarray:
        """
        Computes the partial fugacity of the gas phase for each component.
        
        Args:
            y_i (np.ndarray): The mole fraction of each component in the gas phase.
            
        Returns:
            np.ndarray: The partial fugacity of the gas phase for each component.
        """
        return y_i * self.P_sys
    
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
    
    def get_activity_coefficient(self, x_array):
        """
        Computes the activity coefficient for each component. 
        For a system following Raoult's Law, the activity coefficient is 1.
        
        Returns:
            np.ndarray: The activity coefficient for each component.
        """
        return np.ones(self.num_comp)


        

    
        

