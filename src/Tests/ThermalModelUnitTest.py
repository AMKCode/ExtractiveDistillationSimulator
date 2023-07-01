import unittest
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

from thermo_models.RaoultsLawModel import RaoultsLawModel
from thermo_models.WilsonModel import WilsonModel
from thermo_models.MargulesModel import MargulesModel
import os, sys
import utils.AntoineEquation as AE
import matplotlib.pyplot as plt 
import random as rand


class TestRaoultsLawAntoinePlotting(unittest.TestCase):
    def setUp(self) -> None:
        # Antoine Parameters for benzene
        Ben_A = 4.72583
        Ben_B = 1660.652
        Ben_C = -1.461

        # Antoine Parameters for toluene
        Tol_A = 4.07827
        Tol_B = 1343.943
        Tol_C = -53.773
        
        P_sys = 1.0325
        # Create Antoine equations for benzene and toluene
        benzene_antoine = AE.AntoineEquation(Ben_A, Ben_B, Ben_C)
        toluene_antoine = AE.AntoineEquation(Tol_A, Tol_B, Tol_C)

        # Create a Raoult's law object
        self.TolBenSys = RaoultsLawModel(2,P_sys,[benzene_antoine, toluene_antoine])

    def testPlot(self):
        # Use Raoult's law to plot the Txy
        self.TolBenSys.plot_binary_Txy(100,0)
        
    def testRandomizedConvert_ytox_from_convert_xtoy_output_binary_case(self):
        for i in range(100):
            x1 = rand.random()
            x2 = 1 - x1
            solution = (self.TolBenSys.convert_x_to_y(np.array([x1,x2])))
            y_array_sol = solution[:-1]
            np.testing.assert_allclose(np.array([x1,x2,solution[-1]]), self.TolBenSys.convert_y_to_x(y_array=y_array_sol), atol = 1e-2)
            
            

    

        

class TestMargulesModel(unittest.TestCase):
    # Test a Binary Mixture of Benzene and Toluene
    # Benzene = 1
    # Toluene = 2
    # Constants obtained from Kassmann, et al. (1986)
    def setUp(self) -> None:
        P_sys = 1.0325
        num_comp = 2
        A_ = {
            (1,2) : 0.0092,
            (2,1) : -0.0090
        }

        # Antoine Parameters for benzene
        Ben_A = 4.72583
        Ben_B = 1660.652
        Ben_C = -1.461

        #Antoine Parameters for toluene
        Tol_A = 4.07827
        Tol_B = 1343.943
        Tol_C = -53.773
        
        # Create Antoine equations for benzene and toluene
        benzene_antoine = AE.AntoineEquation(Ben_A, Ben_B, Ben_C)
        toluene_antoine = AE.AntoineEquation(Tol_A, Tol_B, Tol_C)

        #Create a Margules' Model Object
        self.MargulesSys = MargulesModel(num_comp, P_sys, A_, [benzene_antoine, toluene_antoine])

    def testPlot(self):
        # Use Margules's law to plot the Txy
        self.MargulesSys.plot_binary_Txy(100,0)
        
    def test_Convert_ytox_from_convert_xtoy_output_binary_case(self):
        for i in range(100):
            x1 = rand.random()
            x2 = 1 - x1
            solution = (self.MargulesSys.convert_x_to_y(np.array([x1,x2])))
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1,x2,temp_sol]), self.MargulesSys.convert_y_to_x(y_array=y_array_sol),atol=1e-3)



class TestWilsonModel(unittest.TestCase):
    # Test a Ternary Mixture of Acetone, Methyl Acetate, Methanol
    # Acetone = 1
    # Methyl Acetate = 2 (AKA Acetic Acid)
    # Methanol = 3

    def setUp(self) -> None:
        # Lambda values from Prausnitz Table IV
        P_sys = 1.0325
        num_comp = 3
        Lambdas = {
            (1,1) : 1.0,
            (1,2) : 0.5781,
            (1,3) : 0.6917,
            (2,1) : 1.3654,
            (2,2) : 1.0,
            (2,3) : 0.6370,
            (3,1) : 0.7681,
            (3,2) : 0.4871,
            (3,3) : 1.0
            }
        
        #Antoine parameters for Acetone
        Ace_A = 4.42448
        Ace_B = 1312.253
        Ace_C = -32.445

        #Antoine parameters for Methyl Acetate
        MA_A = 4.68206
        MA_B = 1642.54
        MA_C = -39.764

        #Antoine parameters for Methanol
        Me_A = 5.20409
        Me_B = 1581.341
        Me_C = -33.5

        #Antoine Equations 
        Acetate_antoine = AE.AntoineEquation(Ace_A, Ace_B, Ace_C)
        MethylAcetate_antoine = AE.AntoineEquation(MA_A, MA_B, MA_C)
        Methanol_antoine = AE.AntoineEquation(Me_A, Me_B, Me_C)

        # Create a Wilson's Model object
        self.TernarySys = WilsonModel(num_comp,P_sys,Lambdas,[Acetate_antoine, MethylAcetate_antoine, Methanol_antoine] )

    def testPlot(self):
        # Use Wilson Model to plot the Txy
        self.TernarySys.plot_ternary_txy(100,0)

if __name__ == '__main__':
    unittest.main()
