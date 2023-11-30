import unittest
import numpy as np
import os, sys
import csv as csv
import matplotlib.pyplot as plt
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
from thermo_models.RaoultsLawModel import RaoultsLawModel

class TestRaoultsLawAntoineBinaryPlotting(unittest.TestCase):
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
        benzene_antoine = AE.AntoineEquationBase10(Ben_A, Ben_B, Ben_C)
        toluene_antoine = AE.AntoineEquationBase10(Tol_A, Tol_B, Tol_C)

        # Create a Raoult's law object
        self.TolBenSys = RaoultsLawModel(2,P_sys, ["Ben", "Tol"],[benzene_antoine, toluene_antoine])

    def testPlot(self):
        # Use Raoult's law to plot the Txy
        fig, ax = plt.subplots(1,1)
        self.TolBenSys.plot_binary_Txy(100,0,ax)
        plt.show()
        
    def testRandomizedConvert_ytox_from_convert_xtoy_output_binary_case(self):
        for i in range(10000):
            x1, x2 = generate_point_system_random_sum_to_one(2)
            solution = (self.TolBenSys.convert_x_to_y(np.array([x1,x2])))[0]
            y_array_sol = solution[:-1]
            np.testing.assert_allclose(np.array([x1,x2,solution[-1]]), self.TolBenSys.convert_y_to_x(y_array=y_array_sol)[0], atol = 1e-4)
    
    def testRandomizedConvert_xtoy_from_convert_ytox_output_binary_case(self):
        for i in range(10000):
            y1, y2 = generate_point_system_random_sum_to_one(2)
            solution = (self.TolBenSys.convert_y_to_x(np.array([y1,y2])))[0]
            x_array_sol = solution[:-1]
            np.testing.assert_allclose(np.array([y1,y2,solution[-1]]), self.TolBenSys.convert_x_to_y(x_array=x_array_sol)[0], atol = 1e-4)
       

if __name__ == '__main__':
    unittest.main()