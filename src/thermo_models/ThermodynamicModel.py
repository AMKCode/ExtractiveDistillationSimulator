import math as math
import numpy as np
from utils.AntoineEquation import *
from scipy.optimize import fsolve

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

    def __init__(self, num_comp: int, P_sys: float, partial_pressure_eqs=AntoineEquation):
        self.P_sys = P_sys
        self.num_comp = num_comp
        self.partial_pressure_eqs = partial_pressure_eqs
        self.boiling_pts = self.set_boiling_points()

    def convert_x_to_y(self, x_array):
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (list[float]): Liquid mole fraction of each component.

        Raises:
            NotImplementedError: Base class method.
        """

        T_guess = 273 # K

        if self.num_comp == 2:
            # binary case
            def to_solve_x_to_y_binary(var):
                # var[0] = T, var[1] = y1, var[2] = y2
                return [self.get_vapor_pressure(var[0])[0]*self.get_activity_coefficients(x_array)[0]*x_array[0]-self.P_sys*var[1],
                        self.get_vapor_pressure(var[0])[1]*self.get_activity_coefficients(x_array)[1]*x_array[1]-self.P_sys*var[2],
                        var[1]+var[2]-1]

            root = fsolve(to_solve_x_to_y_binary, [T_guess, 0.5, 0.5])
            return root[0], np.array([root[1], root[2]]) # returns the temperature, as well as y_i
        else:
            # ternary case
            def to_solve_x_to_y_ternary(var):
                # var[0] = T, var[1] = y1, var[2] = y2, var[3] = y3
                return [self.get_vapor_pressure(var[0])[0]*self.get_activity_coefficients(x_array)[0]*x_array[0]-self.P_sys*var[1],
                        self.get_vapor_pressure(var[0])[1]*self.get_activity_coefficients(x_array)[1]*x_array[1]-self.P_sys*var[2],
                        self.get_vapor_pressure(var[0])[2]*self.get_activity_coefficients(x_array)[2]*x_array[2]-self.P_sys*var[3],
                        var[1]+var[2]+var[3]-1]
            root = fsolve(to_solve_x_to_y_ternary, [T_guess, 0.3, 0.3, 0.4])
            return root[0], np.array([root[1], root[2], root[3]])

    def convert_y_to_x(self, y_array):
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (np.array): Vapor mole fraction of each component.
        """
        
        T_guess = 273 # K

        if self.num_comp == 2:
            # binary case
            
            def to_solve_y_to_x_binary(var):
                # var[0] = T, var[1] = y1, var[2] = y2
                return [self.get_vapor_pressure(var[0])[0]*self.get_activity_coefficients(var[1:])[0]*var[1]-self.P_sys*y_array[0],
                        self.get_vapor_pressure(var[0])[1]*self.get_activity_coefficients(var[1:])[1]*var[2]-self.P_sys*y_array[1],
                        var[1]+var[2]-1]
            root = fsolve(to_solve_y_to_x_binary, [T_guess, 0.5, 0.5])
            return root[0], np.array([root[1], root[2]]) # returns the temperature, as well as y_i
        
        else:
            # ternary case
            def to_solve_x_to_y_ternary(var):
                # var[0] = T, var[1] = y1, var[2] = y2, var[3] = y3
                return [self.get_vapor_pressure(var[0])[0]*self.get_activity_coefficients(var[1:])[0]*var[1]-self.P_sys*y_array[0],
                        self.get_vapor_pressure(var[0])[1]*self.get_activity_coefficients(var[1:])[1]*var[2]-self.P_sys*y_array[1],
                        self.get_vapor_pressure(var[0])[2]*self.get_activity_coefficients(var[1:])[2]*var[3]-self.P_sys*y_array[2],
                        var[1]+var[2]+var[3]-1]
            root = fsolve(to_solve_x_to_y_ternary, [T_guess, 0.3, 0.3, 0.4])
            return root[0], np.array([root[1], root[2], root[3]])
        
    def get_activity_coefficients(self, x_array):
        """
        Computes the activity coefficients of all of the components given parameters

        Args:
            x_array (np.array): the liquid compositions of the system

        Raises:
        NotImplementedError: Base class method
        """
        raise NotImplementedError("Method get_activity_coefficient not implemented in base class")

    def get_vapor_pressure(self, T:float)->np.ndarray:
        vap_pressure_array = np.zeros(self.num_comp)

        for i in range(len(self.partial_pressure_eqs)):
            vap_pressure_array[i] = self.partial_pressure_eqs[i].get_partial_pressure(T)
        
        return vap_pressure_array
    
    def set_boiling_points(self):
        boiling_pts = np.zeros(self.num_comp)
        for i in range(len(self.partial_pressure_eqs)):
            boiling_pts[i] = self.partial_pressure_eqs[i].get_temperature(self.P_sys)
        return boiling_pts