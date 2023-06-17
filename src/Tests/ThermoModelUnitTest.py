import unittest
import os, sys
import numpy as np
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 

from thermo_models.RaoultsLawModel import RaoultsLawModel
import os, sys
import utils.AntoineEquation as AE
import matplotlib.pyplot as plt 

class TestRaoultsLawModelBinary(unittest.TestCase):
    def setUp(self):
        self.component_vapor_pressure = np.array([10.0, 100.0])  # Example vapor pressures for two components
        self.model = RaoultsLawModel(P_sys = 82)

    def test_convert_x_to_y(self):
        x = np.array([0.2, 0.8])
        expected_y = [0.2 * 10/ 82, 0.8 * 100/82]
        calculated_y = self.model.convert_x_to_y(x, [10,100])
        for i in range(len(expected_y)):
            self.assertAlmostEqual(calculated_y[i], expected_y[i])

    def test_convert_y_to_x(self):
        y = np.array([0.2, 0.8])
        expected_x = np.array([0.4, 0.6])
        calculated_x = self.model.convert_y_to_x(y, [41,109+1/3])
        for i in range(len(expected_x)):
            self.assertAlmostEqual(calculated_x[i], expected_x[i])

class TestRaoultsLawAntoinePlotting(unittest.TestCase):
    def testPlot(self):
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
        raoults_law = RaoultsLawModel(P_sys, [benzene_antoine, toluene_antoine])

        # Use the Antoine equations to find the saturation temperatures
        t_sat_ben = benzene_antoine.get_temperature(P_sys)
        t_sat_tol = toluene_antoine.get_temperature(P_sys)

        # Use Raoult's law to compute the compositions
        T_space = np.linspace(t_sat_ben,t_sat_tol,1000)
        x_Ben, y_Ben = raoults_law.compute_composition(T_space)
        plt.plot(x_Ben,y_Ben) 
        plt.show()
        

if __name__ == '__main__':
    unittest.main()

