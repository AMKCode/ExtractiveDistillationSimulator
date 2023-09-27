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
from thermo_models.WilsonModel import WilsonModel


class TestWilsonModelEthanolWaterAcetone(unittest.TestCase):
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
            self.TernarySys = WilsonModel(num_comp,P_sys,Lambdas,["Etoh", "H2O", "Meth"],[EtOH_antoine, H2O_antoine, Methanol_antoine],False)
            
        def test_RandomConvert_ytox_from_convert_xtoy_output_ternary_case(self):
            print("here")
            for i in range(100):
                print
                x1, x2, x3 = generate_point_system_random_sum_to_one(3)
                
                solution = (self.TernarySys.convert_x_to_y(np.array([x1, x2, x3])))[0]
                y_array_sol = solution[:-1]
                temp_sol = solution[-1]
                np.testing.assert_allclose(np.array([x1, x2, x3, temp_sol]), self.TernarySys.convert_y_to_x(y_array=y_array_sol)[0], atol=1e-4)
                
        # def test_RandomConvert_xtoy_from_convert_ytox_output_ternary_case(self):
        #     rand.seed(0)
        #     for i in range(1):
        #         y1, y2, y3 = generate_point_system_random_sum_to_one(3)
                
        #         solution = (self.TernarySys.convert_y_to_x(np.array([y1, y2, y3])))[0]
        #         x_array_sol = solution[:-1]
        #         temp_sol = solution[-1]
        #         np.testing.assert_allclose(np.array([y1, y2, y3, temp_sol]), self.TernarySys.convert_x_to_y(x_array=x_array_sol)[0], atol=1e-4)
                
        # def testPlot(self):
        # # Use Wilson Model to plot the Txy
        #     boiling_points = [eq.get_boiling_point(self.TernarySys.P_sys) for eq in self.TernarySys.partial_pressure_eqs]
        #     print(boiling_points)
        #     self.TernarySys.plot_ternary_txy(100,0)


if __name__ == '__main__':
    unittest.main()