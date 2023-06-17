import math
import numpy as np
from scipy.optimize import fsolve


class VLEModel:
    def __init__(self,num_comp:int):
        self.num_comp = num_comp
    
    def convert_x_to_y(self, x_array:np.ndarray):
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (list[float]): Liquid mole fraction of each component.

        """ 
        
            
    def convert_y_to_x(self, y_array):
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (list[float]): Vapor mole fraction of each component.

        Raises:
            NotImplementedError: Base class method.
        """
        raise NotImplementedError("Method convert_y_to_x not implemented in base class")
    
    def get_activity_coefficient():
        pass
    
def main():

    