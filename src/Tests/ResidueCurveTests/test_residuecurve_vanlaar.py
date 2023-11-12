import unittest
import numpy as np
import os, sys
import csv as csv
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir, os.pardir)
)

sys.path.append(PROJECT_ROOT) 

import utils.AntoineEquation as AE
from utils.rand_comp_gen import *
from thermo_models.VanLaarModel import *
from utils.plot_csv_soln import *
from distillation.residue_curves import *
  
class TestVanLaar(unittest.TestCase):

    def setUp(self) -> None:
        #Refer to Table A.9 from Knapp Thesis Paper
        A_ij = {
            (1,1):0,
            (1,2):182.0,
            (1,3):795.0,
            (2,1):196,
            (2,2):0,
            (2,3):332.6,
            (3,1):490.0,
            (3,2):163.80,
            (3,3):0
        }
        #Different definition of Antoine where we have to take the negative of B
        Acet_A = 21.3099; Acet_B = 2801.53; Acet_C = -42.875
        Meth_A = 23.4832; Meth_B = 3634.01; Meth_C = -33.768
        
        #Assuming P < 2 atm
        Water_A = 23.2256; Water_B = 3835.18; Water_C = -45.343
        
        #Kanapp Thesis Figure 3.8 uses ln form of Antoine
        AcetoneAntoine = AE.AntoineEquationBaseE(Acet_A,Acet_B,Acet_C)
        MethanolAntoine = AE.AntoineEquationBaseE(Meth_A, Meth_B, Meth_C)
        WaterAntoine = AE.AntoineEquationBaseE(Water_A,Water_B,Water_C)
        
        
        P_sys = 101325
        self.vle_model = VanLaarModel(3, P_sys= P_sys, A_coeff=A_ij,comp_names=["Ace","Meth","Water"], partial_pressure_eqs = [AcetoneAntoine, MethanolAntoine, WaterAntoine])

    def testPlot(self):
        rcm = phase_portraits(thermo_model=self.vle_model)
        fig, ax = plt.subplots(1,1,figsize= (7,7))
        rcm.plot_residue_curve(ax, [0,5],data_points = 100, 
                                   init_comps = [ 
                                                 np.array([0.4,0.4,0.2]),
                                                 np.array([0.6,0.2,0.2]),
                                                 np.array([0.6,0.1,0.3]),
                                                 np.array([0.1,0.8,0.1]),
                                                 np.array([0.1,0.1,0.8])
                                                 ])
        ax.set_title("Residue Curve")
        plt.show()

    
        
if __name__ == '__main__':
    unittest.main()
        