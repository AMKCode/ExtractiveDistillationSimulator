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
from scipy.optimize import brentq


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
        self.s = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
    
    def rectifying_step(self, x_r_j):
        # Fidkowski and Malone, eqn 3b
        r = self.reflux
        return ((r/(r+1))*x_r_j)+((1/(r+1))*self.xD[0])
    
    def stripping_step(self, y_s_j):
        s = self.s
        return ((s/(s+1))*y_s_j)+((1/(s+1))*self.xB[0])
    
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
    
    def find_strip_fixedpoints_binary(self, n):
        rand.seed(0)
        x0_values = []
        y0_values = []
        

        def compute_strip_fixed(y_s):            
            # Check if x_s is zero or very close to zero
            if abs(y_s) < 1e-10:
                return float('inf')
            else:
                sol_array, mesg = self.thermo_model.convert_y_to_x(np.array([y_s, 1 - y_s]))
                x_0 = sol_array[0]
                return x_0 - self.stripping_step(y_s)

        # Define the initial bracket
        a = 0

        # Generate a list of n points in the range [0, 1]
        partition_points = np.linspace(0.0001, 0.9999, n+1)[1:]  # We start from the second point because the first is 0

        for b in partition_points:
            try:
                y_0 = brentq(compute_strip_fixed, a, b, xtol=1e-5)
                x_0 = self.thermo_model.convert_x_to_y(np.array([y_0, 1 - y_0]))[0][0]
                y0_values.append(x_0)
                x0_values.append(y_0)
            except ValueError:
                # If brentq fails because the function has the same sign at the endpoints, we simply skip this partition
                pass
            # Update a to be the current b for the next partition
            a = b

        return x0_values, y0_values

    def find_rect_fixedpoints_binary(self, n):
        rand.seed(0)
        x0_values = []
        y0_values = []

        def compute_rect_fixed(x_r):
            sol_array, mesg = self.thermo_model.convert_x_to_y(np.array([x_r, 1 - x_r]))
            y_x_0 = sol_array[0]
            return - y_x_0 + self.rectifying_step(x_r_j=x_r)

        # Define the initial bracket
        a = 0

        # Generate a list of n points in the range [0, 1]
        partition_points = np.linspace(0, 1, n+1)[1:]  # We start from the second point because the first is 0

        for b in partition_points:
            try:
                x_0 = brentq(compute_rect_fixed, a, b, xtol=1e-5)
                y_0 = self.thermo_model.convert_x_to_y(np.array([x_0, 1- x_0]))[0][0]
                x0_values.append(x_0)
                y0_values.append(y_0)
            except ValueError:
                # If brentq fails because the function has the same sign at the endpoints, we simply skip this partition
                pass
            # Update a to be the current b for the next partition
            a = b

        return x0_values, y0_values

        
    def plot_distil_binary(self):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.") 
        
        x1_space = np.linspace(0, 1, 100)
        x_array = np.column_stack([x1_space, 1 - x1_space])
        y_array, t_evaluated = [], []
        
        for x in x_array:
            solution = self.thermo_model.convert_x_to_y(x)[0]
            y_array.append(solution[:-1])
            t_evaluated.append(solution[-1])
        y_array = np.array(y_array)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
            
        ax.plot(x_array[:,0], y_array[:,0])
        ax.plot([0,1], [0,1], linestyle='dashed')

        x_r = np.linspace(0, 1, 100)
        y_s = np.linspace(0, 1, 100)

        y_r = self.rectifying_step(x_r)
        x_s = self.stripping_step(y_s)
            
        ax.plot(x_r, y_r)
        ax.plot(x_s, y_s)
        
        x_r_0, y_r_0  = self.find_rect_fixedpoints_binary(n=10)
        x_s_0, y_s_0 = self.find_strip_fixedpoints_binary(n=10)
        
        ax.scatter( x_r_0,y_r_0, s=100, c="red")
        ax.scatter(x_s_0, y_s_0, s=100, c="red")
    
        ax.set_aspect('equal', adjustable='box')
        plt.show()

        # def find_fixed_point_binary(self):
    #     def fixed_pt_eqn(var):
    #         # x_0, y_0, r_min
    #         x_0 = var[0]
    #         y_0 = var[1]
    #         r_min = var[2]
    #         q = var[3]
    #         # Fidkowski and Malone eqn 2
    #         s = ((r_min+q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + q - 1
    #         return [y_0 - (((r_min/(r_min+1))*x_0)+((1/(r_min+1))*self.xD[0])), # Fidkowski and Malone eqn 3b
    #                 x_0 - (((s/(s+1))*y_0)+((1/(s+1))*self.xB[0])), # Fidkowski and Malone eqn 4b
    #                 y_0 - self.thermo_model.convert_x_to_y(np.array([x_0, 1-x_0]))[0][0], 
    #                 q - ((s+1)*((self.xD[0]-self.xF[0])/(self.xD[0]-self.xB[0])))+(r_min*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xB[0])))] # Fidkowski and Malone eqn 2
    #     sol = fsolve(fixed_pt_eqn, [0.5, 0.5, 1, 1])
    #     print(sol)
    #     return sol