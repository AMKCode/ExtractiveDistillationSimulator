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
import os, sys
import utils.AntoineEquation as AE
import matplotlib.pyplot as plt 

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
        return
    

        
    def test_Convert_ytox_from_convert_xtoy_output_binary_case(self):
        solution = (self.TolBenSys.convert_x_to_y(np.array([0.5,0.5])))
        y_array_sol = solution[:-1]
        temp_sol = solution[-1]
        assert(solution.all() == self.TolBenSys.convert_y_to_x(y_array=y_array_sol).all())

if __name__ == '__main__':
    unittest.main()

