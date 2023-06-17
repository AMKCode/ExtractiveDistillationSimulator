import math
import numpy as np
from scipy.optimize import fsolve
from ThermoModel import *
from utils.AntoineEquation import *

class IGM_ILM_Model(VLEModel):
    def __init__(self, num_comp: int, P_sys: float, partial_pressure_eqs:np.ndarray):
        super().__init__(num_comp)
        self.P_sys = P_sys
        self.partial_pressure_eqs = partial_pressure_eqs
        
    def compute_gas_partial_fugacity(self,y_i:np.ndarray) -> np.ndarray:
        return y_i * self.P_sys
    
    def get_vapor_pressure(self, Temp:np.ndarray)->np.ndarray:
        vap_pressure_array = []
        for partial_pressure_eq in self.partial_pressure_eqs:
            vap_pressure_array.append(partial_pressure_eq.get_partial_pressure(Temp))
        return np.array(vap_pressure_array)
    
    def get_activity_coefficient(self):
        return np.ones(self.num_comp)
    
    def compute_Txy_eq(self, x:np.ndarray, data_points:int, Temp_range):
        boiling_point = self.partial_pressure_eqs.get_boiling_point()
        Temp_range = np.amax(boiling_point), np.amin(boiling_point)
        Temp_space = np.linspace(Temp_range[0],Temp_range[1], data_points)
        activity_coeff = self.get_activity_coefficient()
        vapor_pressure_array = self.get_vapor_pressure(Temp_space)
        return x * activity_coeff * vapor_pressure_array - self.P_sys
    
    def get_Txy(self):
        pass
        
        

    
        

