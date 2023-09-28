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
from thermo_models.MargulesModel import *
from utils.plot_csv_soln import *
from distillation.residue_curves import *

class TestTernaryMargulesAcetaldehydeMethanolWater_no_jacob(unittest.TestCase):
    
    def setUp(self) -> None:
        A_ = np.array([[0, -316.699, 350.100], [-384.657,0, 307.000],[290.200,143.00,0]])
        P_sys = 1
        
        #Acetaldehyde
        Acet_A = 3.68639
        Acet_B = 822.894
        Acet_C = -69.899
        #Methanol
        Me_A = 5.31301
        Me_B = 1676.569
        Me_C = -21.728
        #Water
        H2O_A = 3.55959
        H2O_B = 643.748
        H2O_C = -198.043
        
        #Antoine Equations 
        Acet_antoine = AE.AntoineEquationBase10(Acet_A, Acet_B, Acet_C)
        H2O_antoine = AE.AntoineEquationBase10(H2O_A, H2O_B, H2O_C)
        Methanol_antoine = AE.AntoineEquationBase10(Me_A, Me_B, Me_C)
        
        self.vle_model = MargulesModelTernary(3,P_sys,A_,["Acet","MeOH","H2O"],[Acet_antoine, Methanol_antoine, H2O_antoine],False)

    def testPlot(self):
        rcm = residue_curve(self.vle_model)
        fig, ax = plt.subplots(1,1,figsize= (7,7))
        rcm.plot_residue_curve_int(ax)
        plt.show()
        
if __name__ == '__main__':
    unittest.main()