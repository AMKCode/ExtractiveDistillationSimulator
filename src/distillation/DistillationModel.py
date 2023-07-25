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
        self.VB = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
    
    def rectifying_step_xtoy(self, x_r_j):
        # Fidkowski and Malone, eqn 3b
        r = self.reflux
        return ((r/(r+1))*x_r_j)+((1/(r+1))*self.xD[0])


    def rectifying_step_ytox(self, y_r_j):
        r = self.reflux
        return (((r+1)/r)*y_r_j - (self.xD[0]/r))
    
    def stripping_step_ytox(self, y_VB_j):
        VB = self.VB
        return ((VB/(VB+1))*y_VB_j)+((1/(VB+1))*self.xB[0])
    
    def stripping_step_xtoy(self, x_VB_j):
        VB = self.VB
        xB = self.xB[0]
        return ((VB+1)/VB)*x_VB_j - (xB/VB)
    
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
        y_VB = np.linspace(0, 1, 100)

        x_0, y_0, r_min, q_ = self.find_fixed_point_binary()

        VB = ((r_min+q_)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + q_ - 1

        y_r = self.rectifying_step(x_r, r_min)
        x_VB = self.stripping_step(y_VB, VB)

        plt.plot(x_r, y_r)
        plt.plot(x_VB, y_VB)
        plt.plot(x_0, y_0, 'ro')
        plt.show()
    
    def find_strip_fixedpoints_binary(self, n):
        rand.seed(0)
        x0_values = []
        y0_values = []
        

        def compute_strip_fixed(x_VB):            
            # Check if x_s is zero or very close to zero
            if abs(x_VB) < 1e-10:
                return float('inf')
            else:
                sol_array, mesg = self.thermo_model.convert_x_to_y(np.array([x_VB, 1 - x_VB]))
                y_0 = sol_array[0]
                return y_0 - self.stripping_step_xtoy(x_VB_j=x_VB)

        # Define the initial bracket
        a = 0

        # Generate a list of n points in the range [0, 1]
        partition_points = np.linspace(0.0001, 0.9999, n+1)[1:]  # We start from the second point because the first is 0

        for b in partition_points:
            try:
                x_VB_0 = brentq(compute_strip_fixed, a, b, xtol=1e-8)
                y_VB_0 = self.thermo_model.convert_x_to_y(np.array([x_VB_0, 1 - x_VB_0]))[0][0]
                y0_values.append(y_VB_0)
                x0_values.append(x_VB_0)
            except ValueError:
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
            return - y_x_0 + self.rectifying_step_xtoy(x_r_j=x_r)

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
                pass
            # Update a to be the current b for the next partition
            a = b
        return x0_values, y0_values

        
    def plot_distil_binary(self, axs):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.") 
        ax1, ax2, ax3 = axs
        ax1.set_xlim([0,1]); ax2.set_xlim([0,1]); ax3.set_xlim([0,1]); 
        ax1.set_ylim([0,1]); ax2.set_ylim([0,1]); ax3.set_ylim([0,1]) 
        
        
        x1_space = np.linspace(0, 1, 1000)
        x_array = np.column_stack([x1_space, 1 - x1_space])
        y_array, t_evaluated = [], []
        
        for x in x_array:
            solution = self.thermo_model.convert_x_to_y(x)[0]
            y_array.append(solution[:-1])
            t_evaluated.append(solution[-1])
        y_array = np.array(y_array)

        ax1.plot(x_array[:,0], y_array[:,0]); ax2.plot(x_array[:,0], y_array[:,0]); ax3.plot(x_array[:,0], y_array[:,0])
        ax1.plot([0,1], [0,1], linestyle='dashed'); ax2.plot([0,1], [0,1], linestyle='dashed'); ax3.plot([0,1], [0,1], linestyle='dashed')

        y_r = self.rectifying_step_xtoy(x1_space)
        y_VB = self.stripping_step_xtoy(x1_space)
        
   
        op_color = 'yellow'
        
        for i in range(len(x1_space)):
            if (abs((y_r[i]) - y_VB[i]) <= 0.001):    
                if (y_r[i] < y_array[i,0]):           
                    op_color = 'green'
                else:
                    op_color = 'red'   
        '''
        ## ADD POINTS TO X AXIS TO REPRESENT NUMBER OF EQUILIBRIA ##
        N = 0
        x1, x2, y1, y2 = self.xD[0], self.xD[0], self.xD[0], self.xD[0]
        x_pts = []
        while (x1 > self.xB[0]):
            x_pts.append(x1)
            N += 1
            solution = self.thermo_model.convert_y_to_x([y1, 1-y1])
            x2 = solution[0]
            y2 = rectifying_step_xtoy(x2)
            x1 = x2
            y1 = y2 

        x_zero = np.zeros(N)       

        ax2.scatter(x_pts, x_zero)
        '''

        ax3.plot(x1_space, y_r, color = op_color); ax2.plot(x1_space, y_r, color = op_color)
        ax3.plot(x1_space, y_VB, color = op_color); ax1.plot(x1_space, y_VB, color = op_color)
        
        x_r_0, y_r_0  = self.find_rect_fixedpoints_binary(n=10)
        x_VB_0, y_VB_0 = self.find_strip_fixedpoints_binary(n=10)
        
        ax3.scatter(x_r_0,y_r_0, s=100, c="red"); ax2.scatter( x_r_0,y_r_0, s=100, c="red")
        ax3.scatter(x_VB_0, y_VB_0, s=100, c="red"); ax1.scatter(x_VB_0, y_VB_0, s=100, c="red")

        ax1.set_xlabel('$x_{1}$'); ax2.set_xlabel('$x_{1}$'); ax3.set_xlabel('$x_{1}$')
        ax1.set_ylabel('$y_{1}$'); ax2.set_ylabel('$y_{1}$'); ax3.set_ylabel('$y_{1}$')
        ax1.set_title("Equilibrium and Stripping Line")
        ax2.set_title("Equilibrium and Rectifying Line")
        ax3.set_title("Equilibrium and Operating Lines")

        ax1.set_aspect('equal', adjustable='box'); ax2.set_aspect('equal', adjustable='box'); ax3.set_aspect('equal', adjustable='box')
        return [ax1,ax2,ax3]

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