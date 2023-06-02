import math as math

class ThermodynamicModel:
    """
    Base class representing a thermodynamic model.

    This is an abstract base class that defines the interface for thermodynamic models.

    Methods:
        convert_x_to_y(x_array):
            Convert liquid mole fraction to vapor mole fraction.

        convert_y_to_x(y_array):
            Convert vapor mole fraction to liquid mole fraction.
    """
    def convert_x_to_y(self, x_array):
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (list[float]): Liquid mole fraction of each component.

        Raises:
            NotImplementedError: Base class method.
        """
        raise NotImplementedError("Method convert_x_to_y not implemented in base class")

    def convert_y_to_x(self, y_array):
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (list[float]): Vapor mole fraction of each component.

        Raises:
            NotImplementedError: Base class method.
        """
        raise NotImplementedError("Method convert_y_to_x not implemented in base class")


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
    
class Antoine_equation:
    def __init__(self,A,B,C):
        self.A = A
        self.B = B
        self.C = C
        
    def partial_pressure(self,Temp):
        return 10**(self.A - self.B/(Temp + self.C))
            
            
        