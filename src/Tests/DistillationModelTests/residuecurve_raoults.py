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
from thermo_models.RaoultsLawModel import *
from utils.plot_csv_soln import *
from distillation.residue_curves import *

class TestTernaryMargulesAcetaldehydeMethanolWater_no_jacob(unittest.TestCase):
    
    def setUp(self) -> None:
        A_ = np.array([[0, -316.699, 350.100], [-384.657,0, 307.000],[290.200,143.00,0]])
        P_sys = 1
        
        # Antoine Parameters for benzene
        Ben_A = 4.72583
        Ben_B = 1660.652
        Ben_C = -1.461

        # Antoine Parameters for toluene
        Tol_A = 4.07827
        Tol_B = 1343.943
        Tol_C = -53.773
        
        # Antoine Parameters for Xylene
        Xyl_A = 4.14553
        Xyl_B = 1474.403
        Xyl_C = -55.377
        
        P_sys = 1.0325
        # Create Antoine equations for benzene and toluene
        benzene_antoine = AE.AntoineEquationBase10(Ben_A, Ben_B, Ben_C)
        toluene_antoine = AE.AntoineEquationBase10(Tol_A, Tol_B, Tol_C)
        xylene_antoine = AE.AntoineEquationBase10(Xyl_A, Xyl_B, Xyl_C)

        # Create a Raoult's law object
        self.vle_model = RaoultsLawModel(3,P_sys,["Ben","Tol","Xyl"],[benzene_antoine, toluene_antoine, xylene_antoine])

    def testPlot(self):
        rcm = residue_curve(self.vle_model)
        fig, ax = plt.subplots(1,1,figsize= (7,7))
        rcm.plot_residue_curve_int(ax, data_points = None, 
                                   init_comps = [ 
                                                 np.array([0.4,0.4,0.2]),
                                                 np.array([0.6,0.2,0.2]),
                                                 np.array([0.6,0.1,0.3]),
                                                 np.array([0.1,0.8,0.1]),
                                                 np.array([0.1,0.1,0.8])
                                                 ])
        plt.show()
        
if __name__ == '__main__':
    unittest.main()