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
from thermo_models.VLEModelBaseClass import *
from thermo_models.RaoultsLawModel import RaoultsLawModel
from thermo_models.WilsonModel import WilsonModel
from thermo_models.MargulesModel import MargulesModel
from thermo_models.VanLaarModel import VanLaarModel
import utils.AntoineEquation as AE
import matplotlib.pyplot as plt 
import random as rand
from scipy.optimize import fsolve


#Notes:
#Conditions for a feasible column, profiles match at the feed stage  + no pinch point in between xB and xD
class DistillationModel:
    def __init__(self, thermo_model:VLEModel, xF = None, xD = None, xB = None, reflux = None, boil_up = None, total_stages = None, feed_stage = None) -> None:
        self.thermo_model = thermo_model
        self.num_comp = thermo_model.num_comp
        self.xF = xF
        self.xD = xD
        self.xB = xB
        self.reflux = reflux
        self.boil_up = boil_up
        self.total_stages = total_stages
        self.feed_stage = feed_stage

        # Fidkowski and Malone, eqn 2
        self.q = ((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0])))-(reflux*((xF[0]-xB[0])/(xD[0]-xB[0])))
    
    def rectifying_step(self, x_r_j, r):
        # Fidkowski and Malone, eqn 3b
        return ((r/(r+1))*x_r_j)+((1/(r+1))*self.xD[0])
    
    def stripping_step(self, y_s_j, s):
        # Fidkowski and Malone, eqn 4b
        return ((s/(s+1))*y_s_j)+((1/(s+1))*self.xB[0])
    
    def find_fixed_point_binary(self):
        def fixed_pt_eqn(var):
            # x_0, y_0, r_min
            x_0 = var[0]
            y_0 = var[1]
            r_min = var[2]
            q = var[3]
            # Fidkowski and Malone eqn 2
            s = ((r_min+q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + q - 1
            return [y_0 - (((r_min/(r_min+1))*x_0)+((1/(r_min+1))*self.xD[0])), # Fidkowski and Malone eqn 3b
                    x_0 - (((s/(s+1))*y_0)+((1/(s+1))*self.xB[0])), # Fidkowski and Malone eqn 4b
                    y_0 - self.thermo_model.convert_x_to_y(np.array([x_0, 1-x_0]))[0][0], 
                    q - ((s+1)*((self.xD[0]-self.xF[0])/(self.xD[0]-self.xB[0])))+(r_min*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xB[0])))] # Fidkowski and Malone eqn 2
        sol = fsolve(fixed_pt_eqn, [0.5, 0.5, 1, 1])
        print(sol)
        return sol
    
    def plot_r_min_binary(self):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
        
        # compute VLE
        x1_space = np.linspace(0, 1, 100)
        x_array = np.column_stack([x1_space, 1 - x1_space])
        y_array, t_evaluated = [], []
        for x in x_array:
            solution = self.thermo_model.convert_x_to_y(x)[0]
            y_array.append(solution[:-1])
            t_evaluated.append(solution[-1])
        y_array = np.array(y_array)

        plt.plot(x_array[:,0], y_array[:,0])
        plt.plot([0,1], [0,1], linestyle='dashed')


        x_r = np.linspace(0, 1, 100)
        y_s = np.linspace(0, 1, 100)

        x_0, y_0, r_min, q_ = self.find_fixed_point_binary()

        s = ((r_min+q_)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + q_ - 1

        y_r = self.rectifying_step(x_r, r_min)
        x_s = self.stripping_step(y_s, s)

        plt.plot(x_r, y_r)
        plt.plot(x_s, y_s)
        plt.plot(x_0, y_0, 'ro')
        plt.show()
    
    
        
        
    
        