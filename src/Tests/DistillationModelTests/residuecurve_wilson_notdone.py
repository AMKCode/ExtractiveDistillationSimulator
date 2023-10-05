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
from thermo_models.WilsonModel import *
from utils.plot_csv_soln import *
from distillation.residue_curves import *
  
class TestVanLaar(unittest.TestCase):

    def setUp(self) -> None:
        #Wilson Constants (Aij) for the Ethanol-Water-Acetone Mixture from Slack
        #1: Ethanol, 2: Water, 3: Acetone
        #Note that the paper uses the e base form of antoine equation with different signs for the addition term (see appendix of paper)
        #Hence, we changed the B term of the antoine constant to its negative
            P_sys = 101325 #1 atm in pascals
            num_comp = 3
            Lambdas = {
                (1,1) : 1,
                (1,2) : 0.1782,
                (1,3) : 0.692,
                (2,1) : 0.8966,
                (2,2) : 1,
                (2,3) : 0.492,
                (3,1) : 0.726,
                (3,2) : 0.066,
                (3,3) : 1
                }
            
            #Antoine parameters for Ethanol
            EtOH_A = 23.5807
            EtOH_B = 3673.81
            EtOH_C = -46.681

            #Antoine parameters for Water
            H2O_A = 23.2256
            H2O_B = 3835.18
            H2O_C = -45.343

            #Antoine parameters for Methanol
            Me_A = 23.4832
            Me_B = 3634.01
            Me_C = -33.768

            #Antoine Equations 
            EtOH_antoine = AE.AntoineEquationBaseE(EtOH_A, EtOH_B, EtOH_C)
            H2O_antoine = AE.AntoineEquationBaseE(H2O_A, H2O_B, H2O_C)
            Methanol_antoine = AE.AntoineEquationBaseE(Me_A, Me_B, Me_C)

            # Create a Wilson's Model object
            self.vle_model = WilsonModel(num_comp,P_sys,Lambdas,["Etoh", "H2O", "Meth"],[EtOH_antoine, H2O_antoine, Methanol_antoine],False)

    def testPlot(self):
        rcm = residue_curve(self.vle_model)
        fig, ax = plt.subplots(1,1,figsize= (7,7))
        rcm.plot_residue_curve_int(ax, [0,3],data_points = 100, 
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
        