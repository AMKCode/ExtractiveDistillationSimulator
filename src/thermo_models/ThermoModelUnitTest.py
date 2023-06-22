import unittest
import numpy as np
from RaoultsLawModel import RaoultsLawModel
import os, sys
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from utils.AntoineEquation import *
    
class TestPlottingBinaryModel(unittest.TestCase):
    def setUp(self) -> None:
        #Antoine Parameters for benzene
        self.Ben_A = 4.72583
        self.Ben_B = 1660.652
        self.Ben_C = -1.461

        #Antoine Parameters for toluene
        self.Tol_A = 4.07827
        self.Tol_B = 1343.943
        self.Tol_C = -53.773
        
        return super().setUp()
    
    def test_plotRaoults(self):
        AntoineEquations = AntoineEquation(self.Ben_A,self.Ben_B,self.Ben_C),AntoineEquation(self.Tol_A,self.Tol_B,self.Tol_C)
        #Total Pressure of system
        P = 1 #bar
        TolBenSys = RaoultsLawModel(2,P,AntoineEquations)
        TolBenSys.plot_binary_Txy(1000,1)

if __name__ == '__main__':
    unittest.main()

