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
from thermo_models.RaoultsLawModel import RaoultsLawModel

class TestTernaryRaoults(unittest.TestCase):
    def setUp(self) -> None:
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
        self.TolBenXylSys = RaoultsLawModel(3,P_sys,["Ben","Tol","Xyl"],[benzene_antoine, toluene_antoine, xylene_antoine])
        
    # def testPlot(self):
    #     self.TolBenXylSys.plot_ternary_txy(100,0)
        
    def test_RandomConvert_ytox_from_convert_xtoy_output_ternary_case(self):
        for i in range(1000):
            x1,x2,x3 =  generate_point_system_random_sum_to_one(3)
            
            solution = (self.TolBenXylSys.convert_x_to_y(np.array([x1, x2, x3])))[0]
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1, x2, x3, temp_sol]), self.TolBenXylSys.convert_y_to_x(y_array=y_array_sol)[0], atol=1e-4)
            
    def test_RandomConvert_xtoy_from_convert_ytox_output_ternary_case(self):
        for i in range(1000):
            y1, y2, y3 = generate_point_system_random_sum_to_one(3)
            
            solution = (self.TolBenXylSys.convert_y_to_x(np.array([y1, y2, y3])))[0]
            x_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([y1, y2, y3, temp_sol]), self.TolBenXylSys.convert_x_to_y(x_array=x_array_sol)[0], atol=1e-4)

if __name__ == '__main__':
    unittest.main()