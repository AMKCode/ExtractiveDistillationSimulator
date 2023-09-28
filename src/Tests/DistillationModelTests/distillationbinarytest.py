import unittest
import numpy as np
import os, sys
import csv as csv
import matplotlib.pyplot as plt
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir, os.pardir)
)

sys.path.append(PROJECT_ROOT) 

from thermo_models.RaoultsLawModel import RaoultsLawModel
from distillation.DistillationBinary import *
import utils.AntoineEquation as AE

class TestDistillationModelBinaryCase(unittest.TestCase):
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
        benzene_antoine = AntoineEquationBase10(Ben_A, Ben_B, Ben_C)
        toluene_antoine = AntoineEquationBase10(Tol_A, Tol_B, Tol_C)

        # Create a Raoult's law object
        vle_model = RaoultsLawModel(2, P_sys, ["Benzene", "Toluene"], [benzene_antoine, toluene_antoine])
        R = 1
        xB = 0.1
        xD = 0.8
        self.distillation_model = DistillationModelBinary(vle_model, xF=np.array([0.5, 0.5]), xD=np.array([xD, 1 - xD]), xB=np.array([xB, 1 - xB]), reflux=R)
    
    def testPlotdistilBinary(self):
        fig, axs = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [40, 1]}, sharex='col')
        self.distillation_model.plot_distil_strip_binary(axs[0],axs[1])
        plt.show()
        
if __name__ == '__main__':
    unittest.main()