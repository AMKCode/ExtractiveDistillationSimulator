from ThermodynamicModel import ThermodynamicModel
import numpy as np

class MargulesModel(ThermodynamicModel):
    """
    A class representing a thermodynamic model based on Margules.

    This model calculates the conversion between liquid mole fraction and vapor mole fraction
    based on Margules's model.

    Args:
        num_comp (int): The number of components of the system -- typically 2 or 3
        P_sys (float): The system pressure in units compatible with the vapor pressures.
        A_dict (dictionary): Dict of floats for {ternary: A12, A21, A23, A32, A13, A31} 



    Methods:
        convert_x_to_y(x, psat, A12, A21):
            Convert liquid mole fraction to vapor mole fraction based on Margules.
        convert_y_to_x(y, psat, A12, A21):
            Convert vapor mole fraction to liquid mole fraction based on Margules.
    """
    def __init__(self,num_comp:int,P_sys:float, A_dict:dict):
        self.num_comp = num_comp
        self.P_sys = P_sys
        self.A_dict = A_dict

    def get_gammas_margules(x_, A_dict):
        #need to update for multicomponent
        gamma1 = np.exp((A12 + 2(A21 - A12)*x_[0]) * (x_[1]**2))
        gamma2 = np.exp((A21 + 2(A12 - A21)*x_[1]) * (x_[0]**2))     
        return np.array([gamma1, gamma2])
    
    #These (x <-> y) functions will be put into base class?
    
    def convert_x_to_y(self, x, psat, A12, A21):
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.
        Args:
            x (np.array): Liquid mole fraction of each component.
            psat (np.array): Saturation pressure of components.
            A12 (float): Margules Parameter
            A21 (float): Margules Parameter
        """
        
        y = np.zeros(x.shape)
        gammas = self.get_gammas_margules(x, A12, A21)
        y = (psat * gammas * x) / self.P_sys
        return y
        
    def convert_y_to_x(self, y, psat, A12, A21):
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.
        Args:
            y (np.array): vapor mole fraction of each component.
            psat (np.array): Saturation pressure of components.
            A12 (float): Margules Parameter
            A21 (float): Margules Parameter
        """
        x = np.zeros(y.shape) + 0.5
        gammas = self.get_gammas_margules(x, A12, A21)
        residuals = self.P_sys*y - psat*gammas*x
        
        while np.abs(np.sum(residuals)) > 0.001: # 0.001 is arbitrarily chosen threshold
            x[0] = (self.P_sys*y[0])/(psat[0]*gammas[0])
            x[1] = 1 - x[0]
        
            gammas = self.get_gammas_margules(x, A12, A21)
            residuals = self.P_sys*y - psat*gammas*x
  
        return x
            