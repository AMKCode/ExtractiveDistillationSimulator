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
from distillation.DistillationTernary import *

class TestVectorField(unittest.TestCase):
    
    def setUp(self) -> None:
        #Acetone (1 - Light) Methanol (2 - Intermediate) Water (3 - Heavy)
        #Table A.6 in Knapp 
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
        # Create a Raoult's law object
        self.vle_model = VanLaarModel(num_comp = 3, P_sys = P_sys, A_coeff = A_ij, comp_names = ["Acetone","Methanol","Water"], partial_pressure_eqs = [AcetoneAntoine, MethanolAntoine, WaterAntoine])


        zF = np.array([0.25, 0.35, 0.4])
        xFL = np.array([0.3, 0.6, 0.1])
        xFU = np.array([0.2, 0.1, 0.7])
        xD = np.array([0.89, 0.05, 0.06]) 
        xB = np.array([0.01, 0.40,0.59])
        R = 5
        Fr = 0.55
        self.distil_model = DistillationModelDoubleFeed(self.vle_model, Fr = Fr, zF = zF, xFL = xFL, xFU = xFU, xD = xD, xB = xB, reflux = R)
    
    def testplotresidue(self):
        rcm = phase_portraits(self.vle_model,self.distil_model)
        fig, ax = plt.subplots(1,1,figsize= (7,7))
        rcm.plot_vector_field_residue(ax,30)
        plt.show()
        
    def testplotmiddle(self):
        rcm = phase_portraits(self.vle_model,self.distil_model)
        fig, ax = plt.subplots(1,1,figsize= (7,7))
        rcm.plot_vector_field_middle(ax,30)
        plt.show()
    
if __name__ == '__main__':
    unittest.main()