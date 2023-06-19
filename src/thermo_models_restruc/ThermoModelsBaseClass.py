import numpy as np
import os, sys
import numpy as np
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 

from utils.AntoineEquation import *
from ThermoModel import *

class IGM_ILM_Model(VLEModel):
    def __init__(self, num_comp: int, P_sys: float, partial_pressure_eqs: AntoineEquation):
        super().__init__(num_comp, P_sys)
        self.partial_pressure_eqs = partial_pressure_eqs
        
    def compute_gas_partial_fugacity(self,y_i:np.ndarray) -> np.ndarray:
        return y_i * self.P_sys
    
    def get_vapor_pressure(self, Temp:float)->np.ndarray:
        vap_pressure_array = []
        for partial_pressure_eq in self.partial_pressure_eqs:
            vap_pressure_array.append(partial_pressure_eq.get_partial_pressure(Temp))
        return np.array(vap_pressure_array)
    
    def get_activity_coefficient(self, *args):
        return np.ones(self.num_comp)
    
def main():
    #Antoine Parameters for benzene
    Ben_A = 4.72583
    Ben_B = 1660.652
    Ben_C = -1.461

    #Antoine Parameters for toluene
    Tol_A = 4.07827
    Tol_B = 1343.943
    Tol_C = -53.773
    
    AntoineEquations = AntoineEquation(Ben_A,Ben_B,Ben_C),AntoineEquation(Tol_A,Tol_B,Tol_C)
    
    # Total Pressure of system
    P = 1 #bar
    
    TolBenSys = IGM_ILM_Model(2,P,AntoineEquations)
    TolBenSys.convert_x_to_y(None)
    
if __name__ == "__main__":
    main()

        

    
        

