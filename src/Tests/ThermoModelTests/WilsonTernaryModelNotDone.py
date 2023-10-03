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
from utils.plot_csv_soln import *


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
            self.vle_model = WilsonModel(num_comp,P_sys,Lambdas,["Etoh", "H2O", "Meth"],[EtOH_antoine, H2O_antoine, Methanol_antoine],False)
            

        def test_RandomConvert_ytox_from_convert_xtoy_output_ternary_case(self):
            passed_path = './src/Tests/test_results/WilsonTernary/convert_ytox_from_convert_xtoy_passed.csv'
            failed_path = './src/Tests/test_results/WilsonTernary/convert_ytox_from_convert_xtoy_failed.csv'
            plot_path = './src/Tests/test_results/WilsonTernary/convert_ytox_from_convert_xtoy_graph.png'
            
            everfailed = False
            with open(passed_path, 'w', newline='') as csvfile, \
                open(failed_path, 'w', newline='') as failed_csvfile:
                    
                passedwriter = csv.writer(csvfile)
                failedwriter = csv.writer(failed_csvfile)
                
                # Write the header row
                passedwriter.writerow(['Iteration', 'x1', 'x2','x3', 'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
                failedwriter.writerow(['Iteration', 'x1', 'x2','x3' ,'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
        
                for i in range(10000):
                    
                    x1,x2,x3 = generate_point_system_random_sum_to_one(self.vle_model.num_comp)
                    print("1st:", i, "xvec:", x1,x2,x3)
                    solution = (self.vle_model.convert_x_to_y(np.array([x1, x2, x3])))[0]
                    y_array_sol = solution[:-1]
                    temp_sol = solution[-1]
                    
                    converted_solution = self.vle_model.convert_y_to_x(y_array=y_array_sol)[0]
                    converted_y = converted_solution[:-1]
                    converted_temp = converted_solution[-1]

                    try:
                        np.testing.assert_allclose(np.array([x1, x2, x3, temp_sol]), converted_solution, atol=1e-4)
                        # Write the results to the CSV file
                        passedwriter.writerow([i, x1, x2, x3, y_array_sol[0], y_array_sol[1],y_array_sol[2],temp_sol, converted_y[0], converted_y[1], converted_y[2], converted_temp])
                    except:
                        failedwriter.writerow([i, x1, x2, x3,y_array_sol[0], y_array_sol[1],y_array_sol[2], temp_sol, converted_y[0], converted_y[1], converted_y[2], converted_temp])
                        everfailed = True
            plot_csv_data(passed_csv_path=passed_path,failed_csv_path=failed_path, labels=["x1", "x2"],plot_path=plot_path)    
                
            if everfailed == True:
                raise AssertionError

        def test_RandomConvert_xtoy_from_convert_ytox_output_ternary_case(self):
            passed_path = './src/Tests/test_results/WilsonTernary/convert_xtoy_from_convert_ytox_passed.csv'
            failed_path = './src/Tests/test_results/WilsonTernary/convert_xtoy_from_convert_ytox_failed.csv'
            plot_path = './src/Tests/test_results/WilsonTernary/convert_xtoy_from_convert_ytox_graph.png'
            
            everfailed = False
            with open(passed_path, 'w', newline='') as csvfile, \
                open(failed_path, 'w', newline='') as failed_csvfile:
                    
                passedwriter = csv.writer(csvfile)
                failedwriter = csv.writer(failed_csvfile)
                
                # Write the header row
                passedwriter.writerow(['Iteration', 'x1', 'x2','x3', 'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
                failedwriter.writerow(['Iteration', 'x1', 'x2','x3' ,'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
            
                for i in range(10000):
                    
                    y1,y2,y3 = generate_point_system_random_sum_to_one(self.vle_model.num_comp)
                    print("2nd:", i, "input:", y1,y2,y3)
                    solution = (self.vle_model.convert_y_to_x(np.array([y1, y2, y3])))[0]
                    x_array_sol = solution[:-1]
                    temp_sol = solution[-1]
                    
                    converted_solution = self.vle_model.convert_x_to_y(x_array=x_array_sol)[0]
                    converted_y = converted_solution[:-1]
                    converted_temp = converted_solution[-1]

                    try:
                        np.testing.assert_allclose(np.array([y1, y2, y3, temp_sol]), converted_solution, atol=1e-4)
                        # Write the results to the CSV file
                        passedwriter.writerow([i,x_array_sol[0], x_array_sol[1],x_array_sol[2], y1, y2, y3, temp_sol, converted_y[0], converted_y[1], converted_y[2], converted_temp])
                    except:
                        failedwriter.writerow([i,x_array_sol[0], x_array_sol[1],x_array_sol[2], y1, y2, y3, temp_sol, converted_y[0], converted_y[1], converted_y[2], converted_temp])
                        everfailed = True
            plot_csv_data(passed_csv_path=passed_path,failed_csv_path=failed_path, labels=["y1", "y2"],plot_path=plot_path)    
                
            if everfailed == True:
                raise AssertionError
        # def testPlot(self):
        # # Use Wilson Model to plot the Txy
        #     boiling_points = [eq.get_boiling_point(self.TernarySys.P_sys) for eq in self.TernarySys.partial_pressure_eqs]
        #     print(boiling_points)
        #     self.TernarySys.plot_ternary_txy(100,0)


if __name__ == '__main__':
    unittest.main()