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
  
class TestVanLaar(unittest.TestCase):
    #Test a binary mixture for vanlar of acetone and water based on
    #https://en.wikipedia.org/wiki/Van_Laar_equation#cite_note-:0-3 
    def setUp(self) -> None:
        A12 = 2.1041
        A21 = 1.5555
        #Referencing NIST website
        Water_A = 3.55959
        Water_B = 643.748
        Water_C = -198.043
        Acet_A = 4.42448
        Acet_B = 1312.253
        Acet_C = -32.445
        
        P_sys = 1
        WaterAntoine = AE.AntoineEquationBase10(Water_A,Water_B,Water_C)
        AcetoneAntoine = AE.AntoineEquationBase10(Acet_A,Acet_B,Acet_C)
        self.AcetWaterVanLaar = VanLaarModel(2,P_sys,[WaterAntoine,AcetoneAntoine],A12,A21)
        
    # def testPlot(self):
    #     self.AcetWaterVanLaar.plot_binary_Txy(100,0)
   
    
    def test_RandomizedConvert_ytox_from_convert_xtoy_output_binary_case(self):
        rand.seed(0)
        for i in range(100):
            x1 = rand.random()
            x2 = 1 - x1
            solution = (self.AcetWaterVanLaar.convert_x_to_y(np.array([x1,x2])))[0]
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1,x2,temp_sol]), self.AcetWaterVanLaar.convert_y_to_x(y_array=y_array_sol)[0],atol=1e-3)
    
    def testRandomizedConvert_xtoy_from_convert_ytox_output_binary_case(self):
        rand.seed(0)
        for i in range(1000):
            y1 = rand.random()
            y2 = 1 - y1
            solution = (self.AcetWaterVanLaar.convert_y_to_x(np.array([y1,y2])))[0]
            x_array_sol = solution[:-1]
            
            expected_array = np.array([y1,y2,solution[-1]])
            actual_array, mesg = self.AcetWaterVanLaar.convert_x_to_y(x_array=x_array_sol)
            if not np.allclose(expected_array, actual_array, atol=1e-4):
                print("\ntestRandomizedConvert_xtoy_from_convert_ytox_output_binary_case")
                print(f"For iteration {i}: Expected array: {expected_array}, but got: {actual_array} with err\n, {mesg}")
                raise ValueError

if __name__ == '__main__':
    unittest.main()