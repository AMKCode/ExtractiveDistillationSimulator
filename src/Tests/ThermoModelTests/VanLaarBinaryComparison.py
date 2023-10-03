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

class TestTernaryRaoults(unittest.TestCase):
    def setUp(self) -> None:
        A12 = 2.1041
        A21 = 1.5555
        
        A_coeff = {
            (1,1):0,
            (1,2): A12,
            (2,1): A21,
            (2,2): 0
        }
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
        
        self.binaryVanlaar = VanLaarBinaryModel(2,P_sys,["h2o","acet"],[WaterAntoine,AcetoneAntoine],A12,A21)
        
        self.ternaryVanlaar = VanLaarModel(2, P_sys,["h2o","acetone"], [WaterAntoine, AcetoneAntoine],A_coeff)
        

        

    def test_RandomConvert_ytox_from_convert_xtoy_binaryvanlaar(self):
        for i in range(100):
            x1,x2 =  generate_point_system_random_sum_to_one(2)
            
            solution = (self.binaryVanlaar.convert_x_to_y(np.array([x1, x2])))[0]
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1, x2, temp_sol]), self.binaryVanlaar.convert_y_to_x(y_array=y_array_sol)[0], atol=1e-4)
            
    def test_RandomConvert_xtoy_from_convert_ytox_output_binaryvanlaar(self):
        for i in range(100):
            y1, y2 = generate_point_system_random_sum_to_one(2)
            
            solution = (self.binaryVanlaar.convert_y_to_x(np.array([y1, y2])))[0]
            x_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([y1, y2, temp_sol]), self.binaryVanlaar.convert_x_to_y(x_array=x_array_sol)[0], atol=1e-4)
            
        
    def test_RandomConvert_ytox_from_convert_xtoy_ternaryvanlaar(self):
        for i in range(100):
            x1,x2 =  generate_point_system_random_sum_to_one(2)
            
            solution = (self.ternaryVanlaar.convert_x_to_y(np.array([x1, x2])))[0]
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1, x2, temp_sol]), self.ternaryVanlaar.convert_y_to_x(y_array=y_array_sol)[0], atol=1e-4)
            
    def test_RandomConvert_xtoy_from_convert_ytox_output_ternaryvanlaar(self):
        for i in range(100):
            y1, y2 = generate_point_system_random_sum_to_one(2)
            
            solution = (self.ternaryVanlaar.convert_y_to_x(np.array([y1, y2])))[0]
            x_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([y1, y2, temp_sol]), self.ternaryVanlaar.convert_x_to_y(x_array=x_array_sol)[0], atol=1e-4)

#Test for equivalence between the two
    
    def test_convertytox_equivalence(self):
        for i in range(1000):
            y1, y2 = generate_point_system_random_sum_to_one(2)
            solution_binary = (self.binaryVanlaar.convert_y_to_x(np.array([y1, y2])))[0]
            solution_ternary = (self.ternaryVanlaar.convert_y_to_x(np.array([y1, y2])))[0]
            
            np.testing.assert_allclose(solution_binary,solution_ternary, atol=1e-4)
            
    def test_convertxtoy_equivalence(self):
        for i in range(1000):
            y1, y2 = generate_point_system_random_sum_to_one(2)
            solution_binary = (self.binaryVanlaar.convert_x_to_y(np.array([y1, y2])))[0]
            solution_ternary = (self.ternaryVanlaar.convert_x_to_y(np.array([y1, y2])))[0]
            
            np.testing.assert_allclose(solution_binary,solution_ternary, atol=1e-4)
    
    def test_plot_compare(self):
        fig, ax = plt.subplots(1,2,figsize=(10, 6))
        self.binaryVanlaar.plot_binary_Txy(100, 0, ax[0])
        self.ternaryVanlaar.plot_binary_Txy(100,0,ax[1])
        plt.show()
        
            
            

if __name__ == '__main__':
    unittest.main()