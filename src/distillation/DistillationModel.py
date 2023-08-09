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
from thermo_models.VLEModelBaseClass import *
import matplotlib.pyplot as plt 
import random as rand
from scipy.optimize import fsolve
from scipy.optimize import brentq
from utils.AntoineEquation import *
from thermo_models.RaoultsLawModel import *


#Notes:
#Conditions for a feasible column, profiles match at the feed stage  + no pinch point in between xB and xD
class DistillationModel:
    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        """
        DistillationModel constructor

        Args:
            thermo_model (VLEModel): Vapor-Liquid Equilibrium (VLE) model to be used in the distillation process.
            xF (np.ndarray): Mole fraction of each component in the feed.
            xD (np.ndarray): Mole fraction of each component in the distillate.
            xB (np.ndarray): Mole fraction of each component in the bottom product.
            reflux (Optional): Reflux ratio. If not provided, it will be calculated based on other parameters.
            boil_up (Optional): Boil-up ratio. If not provided, it will be calculated based on other parameters.
            q (float, optional): Feed condition (q) where q = 1 represents saturated liquid feed and q = 0 represents saturated vapor feed. Defaults to 1.
        
        Raises:
            ValueError: If the reflux, boil-up and q are not correctly specified. Only two of these parameters can be independently set.
        """
        self.thermo_model = thermo_model
        self.num_comp = thermo_model.num_comp
        self.xF = xF
        self.xD = xD
        self.xB = xB
        self.q = q
        
        if reflux is not None and boil_up is not None and q is None:
            self.boil_up = boil_up
            self.reflux = reflux
            self.q = ((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0])))-(reflux*((xF[0]-xB[0])/(xD[0]-xB[0]))) #this one need 1 component
        elif reflux is None and boil_up is not None and q is not None:
            self.boil_up = boil_up
            self.q = q
            self.reflux = (((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0]))) - self.q)/((xF[0]-xB[0])/(xD[0]-xB[0])) #this one need 1 component
        elif reflux is not None and boil_up is None and q is not None:
            self.reflux = reflux
            self.q = q
            self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1 #this one need 1 component
        else:
            raise ValueError("Underspecification or overspecification: only 2 variables between reflux, boil up, and q can be provided")
        # self.x_array_equib, self.y_array_equib, self.t_array = self.compute_equib() 
        
        # # Initialize numpy arrays
        # y_s_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))
        # y_r_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))

        # for i, x1 in enumerate(self.x_array_equib):
        #     y_s_array[i] = self.stripping_step_xtoy(x1)
        #     y_r_array[i] = self.rectifying_step_xtoy(x1)
        
        # self.y_s_array = y_s_array
        # self.y_r_array = y_r_array
        
    def rectifying_step_xtoy(self, x_r_j:np.ndarray):
        """
        Method to calculate y in the rectifying section of the distillation column from given x.

        Args:
            x_r_j (np.ndarray): Mole fraction of each component in the liquid phase in the rectifying section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        r = self.reflux
        xD = self.xD
        return ((r/(r+1))*x_r_j)+((1/(r+1))*xD)

    def rectifying_step_ytox(self, y_r_j):
        """
        Method to calculate x in the rectifying section of the distillation column from given y.

        Args:
            y_r_j (float): Mole fraction of each component in the vapor phase in the rectifying section.

        Returns:
            float: Mole fraction of each component in the liquid phase in the rectifying section which corresponds to y_r_j.
        """
        r = self.reflux
        xD = self.xD
        return (((r+1)/r)*y_r_j - (xD/r))
    
    def stripping_step_ytox(self, y_s_j):
        """
        Method to calculate x in the stripping section of the distillation column from given y.

        Args:
            y_s_j (float): Mole fraction of each component in the vapor phase in the stripping section.

        Returns:
            float: Mole fraction of each component in the liquid phase in the stripping section that corresponds to y_s_j.
        """
        boil_up = self.boil_up
        xB = self.xB
        return ((boil_up/(boil_up+1))*y_s_j)+((1/(boil_up+1))*xB)
    
    def stripping_step_xtoy(self, x_s_j):
        """
        Method to calculate y in the stripping section of the distillation column from given x.

        Args:
            x_s_j (float): Mole fraction of each component in the liquid phase in the stripping section.

        Returns:
            float: Mole fraction of each component in the vapor phase in the stripping section.
        """
        boil_up = self.boil_up
        xB = self.xB
        return ((boil_up+1)/boil_up)*x_s_j - (xB/boil_up)
    
    def compute_equib(self):
        x1_space = np.linspace(0, 1, 1000)
        y_array = np.zeros((x1_space.size, 2))
        t_array = np.zeros(x1_space.size)
        
        # Initialize numpy arrays
        x_array = np.zeros((x1_space.size, 2))
        for i, x1 in enumerate(x1_space):
            x_array[i] = [x1, 1 - x1]  # Fill the x_array directly
            solution = self.thermo_model.convert_x_to_y(x_array[i])[0]
            y_array[i] = solution[:-1]
            t_array[i] = solution[-1]
        return x_array, y_array, t_array
        
    def plot_distil_ternary(self):
        if self.num_comp != 3:
            raise ValueError("This method can only be used for binary distillation.")
    
    def change_r(self, new_r):
        self.reflux = new_r
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
        return self