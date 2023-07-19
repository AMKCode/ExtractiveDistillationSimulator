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
from thermo_models.WilsonModel import WilsonModel
from thermo_models.MargulesModel import MargulesModel
from thermo_models.VanLaarModel import VanLaarModel
from distillation.DistillationModel import *
import utils.AntoineEquation as AE
import matplotlib.pyplot as plt 
import random as rand

class TestDistillationModelBinaryCase(unittest.TestCase):
    def setUp(self) -> None:
        Hex_A = 4.00266
        Hex_B = 1171.53
        Hex_C = -48.784

        Hep_A = 4.02832
        Hep_B = 1268.636
        Hep_C = - 56.199

        P_sys = 1
        hexane_antoine = AE.AntoineEquation (Hex_A, Hex_B, Hex_C)
        heptane_antoine = AE.AntoineEquation (Hep_A, Hep_B, Hep_C)
        self.thermo_model = RaoultsLawModel (2,P_sys, [hexane_antoine, heptane_antoine])
    
    def test_PlotRmin(self):
        dist_model = DistillationModel(self.thermo_model, [0.5, 0.5], [0.9, 0.1], [0.01, 0.99], 1, 1, 1, 1)
        dist_model.plot_r_min_binary()
        plt.show()
        
if __name__ == '__main__':
    unittest.main()