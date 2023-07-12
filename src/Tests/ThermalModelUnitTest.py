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
import utils.AntoineEquation as AE
import matplotlib.pyplot as plt 
import random as rand


class TestRaoultsLawAntoineBinaryPlotting(unittest.TestCase):
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

    # def testPlot(self):
    #     # Use Raoult's law to plot the Txy
    #     self.TolBenSys.plot_binary_Txy(100,0)
        
    def testRandomizedConvert_ytox_from_convert_xtoy_output_binary_case(self):
        rand.seed(0)
        for i in range(100):
            x1 = rand.random()
            x2 = 1 - x1
            solution = (self.TolBenSys.convert_x_to_y(np.array([x1,x2])))
            y_array_sol = solution[:-1]
            np.testing.assert_allclose(np.array([x1,x2,solution[-1]]), self.TolBenSys.convert_y_to_x(y_array=y_array_sol), atol = 1e-4)
    
    def testRandomizedConvert_xtoy_from_convert_ytox_output_binary_case(self):
        rand.seed(0)
        for i in range(100):
            y1 = rand.random()
            y2 = 1 - y1
            solution = (self.TolBenSys.convert_y_to_x(np.array([y1,y2])))
            x_array_sol = solution[:-1]
            np.testing.assert_allclose(np.array([y1,y2,solution[-1]]), self.TolBenSys.convert_x_to_y(x_array=x_array_sol), atol = 1e-4)
            
        
class TestMargulesModelBinary(unittest.TestCase):
    # Test a Binary Mixture of Benzene and Toluene
    # Benzene = 1
    # Toluene = 2
    # Constants obtained from Kassmann, et al. (1986)
    def setUp(self) -> None:
        P_sys = 1.0325
        num_comp = 2
        A_ = {
            (1,2) : 0.0092,
            (2,1) : -0.0090
        }

        # Antoine Parameters for benzene
        Ben_A = 4.72583
        Ben_B = 1660.652
        Ben_C = -1.461

        #Antoine Parameters for toluene
        Tol_A = 4.07827
        Tol_B = 1343.943
        Tol_C = -53.773
        
        # Create Antoine equations for benzene and toluene
        benzene_antoine = AE.AntoineEquation(Ben_A, Ben_B, Ben_C)
        toluene_antoine = AE.AntoineEquation(Tol_A, Tol_B, Tol_C)

        #Create a Margules' Model Object
        self.MargulesSys = MargulesModel(num_comp, P_sys, A_, [benzene_antoine, toluene_antoine])

    # def testPlot(self):
    #     # Use Margules's law to plot the Txy
    #     self.MargulesSys.plot_binary_Txy(100,0)
        
    def test_RandomizedConvert_ytox_from_convert_xtoy_output_binary_case(self):
        rand.seed(0)
        for i in range(100):
            x1 = rand.random()
            x2 = 1 - x1
            solution = (self.MargulesSys.convert_x_to_y(np.array([x1,x2])))
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1,x2,temp_sol]), self.MargulesSys.convert_y_to_x(y_array=y_array_sol),atol=1e-4)
            
    def testRandomizedConvert_xtoy_from_convert_ytox_output_binary_case(self):
        rand.seed(0)
        for i in range(100):
            y1 = rand.random()
            y2 = 1 - y1
            solution = (self.MargulesSys.convert_y_to_x(np.array([y1,y2])))
            x_array_sol = solution[:-1]
            np.testing.assert_allclose(np.array([y1,y2,solution[-1]]), self.MargulesSys.convert_x_to_y(x_array=x_array_sol), atol = 1e-4)
            
class TestTernaryRaoults(unittest.TestCase):
    def setUp(self) -> None:
        # Antoine Parameters for benzene
        Ben_A = 4.72583
        Ben_B = 1660.652
        Ben_C = -1.461

        # Antoine Parameters for toluene
        Tol_A = 4.07827
        Tol_B = 1343.943
        Tol_C = -53.773
        
        # Antoine Parameters for Xylene
        Xyl_A = 4.14553
        Xyl_B = 1474.403
        Xyl_C = -55.377
        
        P_sys = 1.0325
        # Create Antoine equations for benzene and toluene
        benzene_antoine = AE.AntoineEquation(Ben_A, Ben_B, Ben_C)
        toluene_antoine = AE.AntoineEquation(Tol_A, Tol_B, Tol_C)
        xylene_antoine = AE.AntoineEquation(Xyl_A, Xyl_B, Xyl_C)

        # Create a Raoult's law object
        self.TolBenXylSys = RaoultsLawModel(3,P_sys,[benzene_antoine, toluene_antoine, xylene_antoine])
        
    # def testPlot(self):
    #     self.TolBenXylSys.plot_ternary_txy(100,0)
        
    def test_RandomConvert_ytox_from_convert_xtoy_output_ternary_case(self):
        rand.seed(0)
        for i in range(100):
            x1 = rand.uniform(0,1)
            x2 = rand.uniform(0,1 - x1)
            x3 = 1 - (x1 + x2)
            
            solution = (self.TolBenXylSys.convert_x_to_y(np.array([x1, x2, x3])))
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1, x2, x3, temp_sol]), self.TolBenXylSys.convert_y_to_x(y_array=y_array_sol), atol=1e-4)
            
    def test_RandomConvert_xtoy_from_convert_ytox_output_ternary_case(self):
        rand.seed(0)
        for i in range(100):
            y1 = rand.uniform(0,1)
            y2 = rand.uniform(0,1 - y1)
            y3 = 1 - (y1 + y2)
            
            solution = (self.TolBenXylSys.convert_y_to_x(np.array([y1, y2, y3])))
            x_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([y1, y2, y3, temp_sol]), self.TolBenXylSys.convert_x_to_y(x_array=x_array_sol), atol=1e-4)

class TestVanLaar(unittest.TestCase):
    #Test a binary mixture for vanlar of acetone and water based on
    #https://en.wikipedia.org/wiki/Van_Laar_equation#cite_note-:0-3 
    def setUp(self) -> None:
        A12 = 2.1041
        A21 = 1.5555
        #Referencing NIST website
        Water_A = 3.55959
        Water_B = 643.748
        Water_C = -198.043
        Acet_A = 4.42448
        Acet_B = 1312.253
        Acet_C = -32.445
        
        P_sys = 1
        WaterAntoine = AE.AntoineEquation(Water_A,Water_B,Water_C)
        AcetoneAntoine = AE.AntoineEquation(Acet_A,Acet_B,Acet_C)
        self.AcetWaterVanLaar = VanLaarModel(2,P_sys,[WaterAntoine,AcetoneAntoine],A12,A21)
        
    # def testPlot(self):
    #     self.AcetWaterVanLaar.plot_binary_Txy(100,0)
   
    
    def test_RandomizedConvert_ytox_from_convert_xtoy_output_binary_case(self):
        rand.seed(0)
        for i in range(100):
            x1 = rand.random()
            x2 = 1 - x1
            solution = (self.AcetWaterVanLaar.convert_x_to_y(np.array([x1,x2])))
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1,x2,temp_sol]), self.AcetWaterVanLaar.convert_y_to_x(y_array=y_array_sol),atol=1e-3)
    
    def testRandomizedConvert_xtoy_from_convert_ytox_output_binary_case(self):
        rand.seed(0)
        for i in range(100):
            y1 = rand.random()
            y2 = 1 - y1
            solution = (self.AcetWaterVanLaar.convert_y_to_x(np.array([y1,y2])))
            x_array_sol = solution[:-1]
            np.testing.assert_allclose(np.array([y1,y2,solution[-1]]), self.AcetWaterVanLaar.convert_x_to_y(x_array=x_array_sol), atol = 1e-4)

    
        
    
        

  
        
class TestWilsonModel(unittest.TestCase):
    # Test a Ternary Mixture of Acetone, Methyl Acetate, Methanol
    # Acetone = 1
    # Methyl Acetate = 2 (AKA Acetic Acid)
    # Methanol = 3

    def setUp(self) -> None:
        # Lambda values from Prausnitz Table IV
        P_sys = 1.0325
        num_comp = 3
        Lambdas = {
            (1,1) : 1.0,
            (1,2) : 0.5781,
            (1,3) : 0.6917,
            (2,1) : 1.3654,
            (2,2) : 1.0,
            (2,3) : 0.6370,
            (3,1) : 0.7681,
            (3,2) : 0.4871,
            (3,3) : 1.0
            }
        
        #Antoine parameters for Acetone
        Ace_A = 4.42448
        Ace_B = 1312.253
        Ace_C = -32.445

        #Antoine parameters for Methyl Acetate
        MA_A = 4.68206
        MA_B = 1642.54
        MA_C = -39.764

        #Antoine parameters for Methanol
        Me_A = 5.20409
        Me_B = 1581.341
        Me_C = -33.5

        #Antoine Equations 
        Acetate_antoine = AE.AntoineEquation(Ace_A, Ace_B, Ace_C)
        MethylAcetate_antoine = AE.AntoineEquation(MA_A, MA_B, MA_C)
        Methanol_antoine = AE.AntoineEquation(Me_A, Me_B, Me_C)

        # Create a Wilson's Model object
        self.TernarySys = WilsonModel(num_comp,P_sys,Lambdas,[Acetate_antoine, MethylAcetate_antoine, Methanol_antoine])
    
    def testSpecific1(self):
        y1 = 0.006698811990524245
        y2 = 0.9086022817212395
        y3 = 0.08469890628823629
        solution = (self.TernarySys.convert_y_to_x(np.array([y1, y2, y3])))
        x_array_sol = solution[:-1]
        temp_sol = solution[-1]
        np.testing.assert_allclose(
            np.array([y1, y2, y3, temp_sol]), 
            self.TernarySys.convert_x_to_y(x_array=x_array_sol), 
            atol=1e-4,
            err_msg=f"Failed for y1={y1}, y2={y2}, y3={y3}, solution={solution}"
            )
        
        solution = (self.TernarySys.convert_y_to_x(np.array([y1, y2, y3])))
        x_array_sol = solution[:-1]
        temp_sol = solution[-1]
        np.testing.assert_allclose(
            np.array([y1, y2, y3, temp_sol]), 
            self.TernarySys.convert_x_to_y(x_array=x_array_sol), 
            atol=1e-4,
            err_msg=f"Failed for y1={y1}, y2={y2}, y3={y3}, solution={solution}"
            )
    
    def testSpecific2(self):
        y1 = 0.07317254860252909
        y2 = 0.8913784438875934
        y3 = 0.03544900750987756
        print(y1+y2+y3)
        solution = (self.TernarySys.convert_y_to_x(np.array([y1, y2, y3])))
        x_array_sol = solution[:-1]
        temp_sol = solution[-1]
        np.testing.assert_allclose(
            np.array([y1, y2, y3, temp_sol]), 
            self.TernarySys.convert_x_to_y(x_array=x_array_sol), 
            atol=1e-4,
            err_msg=f"Sum = {y1+y2+y3} ,Failed for y1={y1}, y2={y2}, y3={y3}, solution={solution}"
            )
        
        solution = (self.TernarySys.convert_y_to_x(np.array([y1, y2, y3])))
        x_array_sol = solution[:-1]
        temp_sol = solution[-1]
        np.testing.assert_allclose(
            np.array([y1, y2, y3, temp_sol]), 
            self.TernarySys.convert_x_to_y(x_array=x_array_sol), 
            atol=1e-4,
            err_msg=f"Failed for y1={y1}, y2={y2}, y3={y3}, solution={solution}"
            )
        

    def test_RandomConvert_ytox_from_convert_xtoy_output_ternary_case(self):
        rand.seed(0)
        for i in range(1000):
            x1 = rand.uniform(0,1)
            x2 = rand.uniform(0,1 - x1)
            x3 = 1 - (x1 + x2)
            
            solution = (self.TernarySys.convert_x_to_y(np.array([x1, x2, x3])))
            y_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(np.array([x1, x2, x3, temp_sol]), 
                self.TernarySys.convert_y_to_x(y_array=y_array_sol),
                atol=1e-4,
                err_msg=f"Failed for x1={x1}, x2={x2}, x3={x3}, solution={solution}"
                )
            
    def test_RandomConvert_xtoy_from_convert_ytox_output_ternary_case(self):
        rand.seed(0)
        for i in range(1000):
            y1 = rand.uniform(0,1)
            y2 = rand.uniform(0,1 - y1)
            y3 = 1 - (y1 + y2)
            
            solution = (self.TernarySys.convert_y_to_x(np.array([y1, y2, y3])))
            x_array_sol = solution[:-1]
            temp_sol = solution[-1]
            np.testing.assert_allclose(
                np.array([y1, y2, y3, temp_sol]), 
                self.TernarySys.convert_x_to_y(x_array=x_array_sol), 
                atol=1e-4,
                err_msg=f"Failed for y1={y1}, y2={y2}, y3={y3}, solution={solution}"
                )

    # def testPlot(self):
        # Use Wilson Model to plot the Txy
        # self.TernarySys.plot_ternary_txy(100,0)

if __name__ == '__main__':
    unittest.main()
