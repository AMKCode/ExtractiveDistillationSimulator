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

    def setUp(self) -> None:
        #Refer to Table A.9 from Knapp Thesis Paper
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
        self.vle_model = VanLaarModel(3, P_sys= P_sys, A_coeff=A_ij,comp_names=["Ace","Meth","Water"], partial_pressure_eqs = [AcetoneAntoine, MethanolAntoine, WaterAntoine], A= None, B=None)
        
    def testPlot(self):
        print('here')
        self.vle_model.plot_ternary_txy(250,1)
   
    # def testConvertx_to_y(self):
    #     # x1,x2,x3 = generate_point_system_random_sum_to_one(3)
    #     x1,x2,x3 = [0.00951773168919824, 0.9725560876952506, 0.017926180615551193]
    #     print("x vector", x1,x2,x3)
    #     print(self.vle_model.get_activity_coefficient(np.array([x1,x2,x3]), 283))
        
    #     solution = (self.vle_model.convert_x_to_y(np.array([x1, x2, x3])))[0]
    #     print(solution)

    def test_RandomConvert_ytox_from_convert_xtoy_output_ternary_case(self):
        passed_path = './src/Tests/test_results/VanLaarTernary/convert_ytox_from_convert_xtoy_passed.csv'
        failed_path = './src/Tests/test_results/VanLaarTernary/convert_ytox_from_convert_xtoy_failed.csv'
        plot_path = './src/Tests/test_results/VanLaarTernary/convert_ytox_from_convert_xtoy_graph.png'
        
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
        passed_path = './src/Tests/test_results/VanLaarTernary/convert_xtoy_from_convert_ytox_passed.csv'
        failed_path = './src/Tests/test_results/VanLaarTernary/convert_xtoy_from_convert_ytox_failed.csv'
        plot_path = './src/Tests/test_results/VanLaarTernary/convert_xtoy_from_convert_ytox_graph.png'
        
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

if __name__ == '__main__':
    unittest.main()