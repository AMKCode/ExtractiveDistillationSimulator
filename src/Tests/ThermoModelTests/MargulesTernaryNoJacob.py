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

class TestTernaryMargulesAcetaldehydeMethanolWater_no_jacob(unittest.TestCase):
    
    def setUp(self) -> None:
        #Margules Parameters from Knapp Thesis
        A_ = np.array([[0, -316.699, 350.100], [-384.657,0, 307.000],[290.200,143.00,0]])
        P_sys = 1.01325 #atm
        
        #Antoine parameters from NIST (Kelvin, Bar)
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
        
        self.vlemodel = MargulesModelTernary(num_comp=3,P_sys=P_sys,A_=A_,comp_names=["Acet","MeOH","H2O"],partial_pressure_eqs=[Acet_antoine, Methanol_antoine, H2O_antoine],use_jacob=False)
        
    def testPlot(self):
        boiling_points = [eq.get_boiling_point(self.vlemodel.P_sys) for eq in self.vlemodel.partial_pressure_eqs]
        print(boiling_points)
        self.vlemodel.plot_ternary_txy(100,0)

    def test_RandomConvert_ytox_from_convert_xtoy_output_ternary_case(self):
        passed_path = './src/Tests/test_results/Margules_acetMethanolWater_no_jacob/convert_ytox_from_convert_xtoy_passed.csv'
        failed_path = './src/Tests/test_results/Margules_acetMethanolWater_no_jacob/convert_ytox_from_convert_xtoy_failed.csv'
        plot_path = './src/Tests/test_results/Margules_acetMethanolWater_no_jacob/convert_ytox_from_convert_xtoy_graph.png'
        
        everfailed = False
        with open(passed_path, 'w', newline='') as csvfile, \
            open(failed_path, 'w', newline='') as failed_csvfile:
                
            passedwriter = csv.writer(csvfile)
            failedwriter = csv.writer(failed_csvfile)
            
            # Write the header row
            passedwriter.writerow(['Iteration', 'x1', 'x2','x3', 'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
            failedwriter.writerow(['Iteration', 'x1', 'x2','x3' ,'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
        
            for i in range(200):
                print(i)
                x1,x2,x3 = generate_point_system_random_sum_to_one(self.vlemodel.num_comp)

                solution = (self.vlemodel.convert_x_to_y(np.array([x1, x2, x3])))[0]
                y_array_sol = solution[:-1]
                temp_sol = solution[-1]
                
                converted_solution = self.vlemodel.convert_y_to_x(y_array=y_array_sol)[0]
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
        passed_path = './src/Tests/test_results/Margules_acetMethanolWater_no_jacob/convert_xtoy_from_convert_ytox_passed.csv'
        failed_path = './src/Tests/test_results/Margules_acetMethanolWater_no_jacob/convert_xtoy_from_convert_ytox_failed.csv'
        plot_path = './src/Tests/test_results/Margules_acetMethanolWater_no_jacob/convert_xtoy_from_convert_ytox_graph.png'
        
        everfailed = False
        with open(passed_path, 'w', newline='') as csvfile, \
            open(failed_path, 'w', newline='') as failed_csvfile:
                
            passedwriter = csv.writer(csvfile)
            failedwriter = csv.writer(failed_csvfile)
            
            # Write the header row
            passedwriter.writerow(['Iteration', 'x1', 'x2','x3', 'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
            failedwriter.writerow(['Iteration', 'x1', 'x2','x3' ,'y1', 'y2','y3', 'Temp', 'Converted_y1', 'Converted_y2', 'Converted_y3', 'Converted_Temp'])
        
            for i in range(200):
                print(i)
                y1,y2,y3 = generate_point_system_random_sum_to_one(self.vlemodel.num_comp)

                solution = (self.vlemodel.convert_y_to_x(np.array([y1, y2, y3])))[0]
                x_array_sol = solution[:-1]
                temp_sol = solution[-1]
                
                converted_solution = self.vlemodel.convert_x_to_y(x_array=x_array_sol)[0]
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