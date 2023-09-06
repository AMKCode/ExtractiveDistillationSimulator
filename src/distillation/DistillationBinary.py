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
import matplotlib.pyplot as plt 
import random as rand
from scipy.optimize import fsolve
from scipy.optimize import brentq
from utils.AntoineEquation import *
from thermo_models.RaoultsLawModel import *
from distillation.DistillationModel import DistillationModel

class DistillationModelBinary(DistillationModel):
    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        super().__init__(thermo_model,xF,xD,xB,reflux,boil_up,q)
        
        self.x_array_equib, self.y_array_equib, self.t_array = self.compute_equib() 
        
        # Initialize numpy arrays
        y_s_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))
        y_r_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))

        for i, x1 in enumerate(self.x_array_equib):
            y_s_array[i] = self.stripping_step_xtoy(x1)
            y_r_array[i] = self.rectifying_step_xtoy(x1)
        
        self.y_s_array = y_s_array
        self.y_r_array = y_r_array
        
        x_r_fixed, y_r_fixed = self.find_rect_fixedpoints_binary(n=30)
        x_s_fixed, y_s_fixed = self.find_strip_fixedpoints_binary(n=30)
        
        self.x_r_fixed = x_r_fixed
        self.y_r_fixed = y_r_fixed
        self.x_s_fixed = x_s_fixed
        self.y_s_fixed = y_s_fixed

        self.x_array_equib, self.y_array_equib, self.t_array = self.compute_equib() 
        
        # Initialize numpy arrays
        y_s_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))
        y_r_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))

        for i, x1 in enumerate(self.x_array_equib):
            y_s_array[i] = self.stripping_step_xtoy(x1)
            y_r_array[i] = self.rectifying_step_xtoy(x1)
        
        self.y_s_array = y_s_array
        self.y_r_array = y_r_array
        
    def find_rect_fixedpoints_binary(self, n):
        rand.seed(0)
        x0_values = []
        y0_values = []

        def compute_rect_fixed(x_r):
            sol_array, mesg = self.thermo_model.convert_x_to_y(np.array([x_r, 1 - x_r]))
            y_x_0 = sol_array[0]
            return - y_x_0 + self.rectifying_step_xtoy(x_r_j=x_r)[0]

        # Define the initial bracket
        a = 0

        # Generate a list of n points in the range [0, 1]
        partition_points = np.linspace(0.0001, 0.9999, n+1)[1:]  # We start from the second point because the first is 0

        for b in partition_points:
            try:
                x_r_0 = brentq(compute_rect_fixed, a, b, xtol=1e-5)
                y_r_0 = self.thermo_model.convert_x_to_y(np.array([x_r_0, 1- x_r_0]))[0][0]
                x0_values.append(x_r_0)
                y0_values.append(y_r_0)
            except ValueError:
                pass
            # Update a to be the current b for the next partition
            a = b
    #    Define the initial bracket
    #     a = 0

    #     # Generate a list of n points in the range [0, 1]
    #     partition_points = np.linspace(0.0001, 0.9999, n+1)[1:]  # We start from the second point because the first is 0

    #     for b in partition_points:
    #         # Define various initial guesses
    #         initial_guesses = [a, b, (a + b) / 2]  # You can add more if necessary

    #         for guess in initial_guesses:
    #             try:
    #                 x_r_0 = fsolve(compute_rect_fixed, guess)[0]
    #                 y_r_0 = self.thermo_model.convert_x_to_y(np.array([x_r_0, 1- x_r_0]))[0][0]
    #                 x0_values.append(x_r_0)
    #                 y0_values.append(y_r_0)
    #                 break  # Break if a solution is found
    #             except ValueError:
    #                 pass
    #             # Update a to be the current b for the next partition
    #             a = b
        return x0_values, y0_values 
    
    
    def find_strip_fixedpoints_binary(self, n):
        rand.seed(0)
        x0_values = []
        y0_values = []
        
        def compute_strip_fixed(x_s):            
            # Check if x_s is zero or very close to zero
            if abs(x_s) < 1e-10:
                return float('inf')
            else:
                sol_array, mesg = self.thermo_model.convert_x_to_y(np.array([x_s, 1 - x_s]))
                y_s_0 = sol_array[0]
                return y_s_0 - self.stripping_step_xtoy(x_s_j = x_s)[0]

        # Define the initial bracket
        a = 0

        # Generate a list of n points in the range [0, 1]
        partition_points = np.linspace(0.0001, 0.9999, n+1)[1:]  # We start from the second point because the first is 0

        for b in partition_points:
            try:
                x_s_0 = brentq(compute_strip_fixed, a, b, xtol=1e-8)
                y_s_0 = self.thermo_model.convert_x_to_y(np.array([x_s_0, 1 - x_s_0]))[0][0]
                y0_values.append(y_s_0)
                x0_values.append(x_s_0)
            except ValueError:
                pass
            # Update a to be the current b for the next partition
            a = b

        return x0_values, y0_values
    
    def plot_distil_strip_binary(self, ax, ax_fixed):
        # Set limits for all main plots
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
            
        #Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])
        
        #Plot the stripping line
        s_min_index = int(1000 * self.xB[0])
        for i in range(len(self.y_s_array)):
            if (self.y_s_array[i,0] > self.y_array_equib[i,0]):
                s_max_index = i
                break 

        ax.plot(self.x_array_equib[s_min_index:s_max_index, 0], self.y_s_array[s_min_index:s_max_index, 0], color = 'green')
        
        #Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')
        
        #Plot the fixed points from stripping line

        ax.scatter(self.x_s_fixed, self.y_s_fixed, s=50, c="red")
        x_fixed, y_fixed, N_1 = self.compute_equib_stages_binary(0, self.x_s_fixed)

        ax.plot(x_fixed, y_fixed, linestyle='--', color='black', alpha = 0.3)
        
        ax.scatter(self.x_s_fixed, self.y_s_fixed, s=50, c="red")

        ax_fixed.set_xlabel('$x_{1}$')
        ax_fixed.xaxis.set_label_coords(0.5, -0.05)
        #ax_fixed.text(0.5, -5, f"Number of Stages: {N_1}", ha='center', va='center', transform=ax_fixed.transAxes)
        ax_fixed.yaxis.set_ticks([])
        ax_fixed.scatter(x_fixed, [0]*len(x_fixed), marker='^', color='blue', facecolors='none', edgecolors='blue', linewidths = 0.75)
        ax.set_aspect('equal', adjustable='box')

        ax_fixed.scatter(self.x_s_fixed, [0]*len(self.x_s_fixed), marker='x', color='black')
        ax_fixed.spines['top'].set_visible(False)
        ax_fixed.spines['right'].set_visible(False)
        ax_fixed.spines['bottom'].set_visible(False)
        ax_fixed.spines['left'].set_visible(False)
        ax_fixed.axhline(0, color='black')  # y=0 line
        ax_fixed.set_xlim([0,1])
        ax_fixed.yaxis.set_ticks([])
        ax_fixed.yaxis.set_ticklabels([])

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_title("Equilibrium and Stripping Line")
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=35)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)

        return ax, ax_fixed
    
    def plot_distil_rect_binary(self, ax, ax_fixed, zoom_factor=0):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
        
        # Set limits for ax
        if (zoom_factor == 0):
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
        else:
            ax.set_xlim([0.72,0.82])
            ax.set_ylim([0.80,0.90])

        #Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])
        
        #Plot the rectifying line
        r_max_index = int(1000 * self.xD[0])
        for i in range(len(self.y_r_array)):
            if (self.y_r_array[i,0] < self.y_array_equib[i,0]):
                r_min_index = i
                break 
        ax.plot(self.x_array_equib[r_min_index:r_max_index, 0], self.y_r_array[r_min_index:r_max_index, 0], color = 'green')
        
        #Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')

        
        x_stages, y_stages, N_2 = self.compute_equib_stages_binary(1, self.x_r_fixed)
        ax.plot(x_stages, y_stages, linestyle='--', color='black', alpha = 0.3)
        ax.scatter(self.x_r_fixed, self.y_r_fixed, s=50, c="red")

        ax_fixed.scatter(x_stages, [0]*len(x_stages), marker='o', color='blue', facecolors='none', edgecolors='blue', linewidths = 0.75)
        #ax_fixed.text(0.5, -5, f"Number of Stages: {N_2}", ha='center', va='center', transform=ax_fixed.transAxes)
        ax_fixed.yaxis.set_ticks([])
        ax.set_aspect('equal', adjustable='box')

        ax_fixed.scatter(self.x_r_fixed, [0]*len(self.x_r_fixed), marker='x', color='black')
        ax_fixed.spines['top'].set_visible(False)
        ax_fixed.spines['right'].set_visible(False)
        ax_fixed.spines['bottom'].set_visible(False)
        ax_fixed.spines['left'].set_visible(False)
        ax_fixed.axhline(0, color='black')  # y=0 line
        if zoom_factor == 0:
            ax_fixed.set_xlim([0,1])
        else:
            ax_fixed.set_xlim([0.72,0.82])
        ax_fixed.yaxis.set_ticks([])
        ax_fixed.yaxis.set_ticklabels([])

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_title("Equilibrium and Rectifying Line")
        ax.set_xlabel(self.thermo_model.comp_names[0])
        ax.set_xlabel(self.thermo_model.comp_names[1])
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=35)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        return ax, ax_fixed
    
    def compute_equib_stages_binary(self, ax_num, fixed_points = []):
        ## ADD POINTS TO X AXIS TO REPRESENT NUMBER OF EQUILIBRIA ##
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")

        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0

        if ax_num == 0:
            N = 0  # Track number of equib stages
            x1 = self.xB[0]
            y1 = self.stripping_step_xtoy(np.array([x1, 1 - x1]))[0]

            while x1 < self.xD[0]:
                if x1 > 1 or y1 > 1 or x1 < 0 or y1 < 0:
                    raise ValueError("Components out of range")

                if np.isclose(x1, fixed_points, rtol=0.001).any():
                    return x_comp, y_comp, "Infinite Stages"
                    
                counter += 1   
                if counter > 200:
                    return x_comp, y_comp, "Too many iterations > 200"

                x_comp.append(x1)
                y_comp.append(y1)
                N += 1

                y2 = self.thermo_model.convert_x_to_y(np.array([x1, 1-x1]))[0][0]
                x_comp.append(x1)
                y_comp.append(y2)

                x2 = self.stripping_step_ytox(np.array([y2, 1 - y2]))[0]  # Step to x value on operating line
                x1, y1 = x2, y2

        elif ax_num in [1, 2]:
            N = 0  # Track number of equib stages
            x1 = self.xD[0]
            y1 = self.rectifying_step_xtoy(np.array([x1, 1 -x1]))[0]

            while x1 > self.xB[0]:
                if x1 > 1 or y1 > 1 or x1 < 0 or y1 < 0:
                    raise ValueError("Components out of range")
                counter += 1
                if np.isclose(x1, fixed_points, atol=0.001).any():
                    return x_comp, y_comp, "Infinite Stages"
                if counter > 200:
                    return x_comp, y_comp, "Too many iterations > 200"

                x_comp.append(x1)
                y_comp.append(y1)
                N += 1

                x2 = self.thermo_model.convert_y_to_x(np.array([y1, 1-y1]))[0][0]
                x_comp.append(x2)
                y_comp.append(y1)

                yr = self.rectifying_step_xtoy(np.array([x2, 1 - x2]))[0]
                ys = self.stripping_step_xtoy(np.array([x2, 1 - x2]))[0]
                y2 = min(yr, ys)
                x1 = x2
                counter += 1

                y1 = yr if ax_num == 1 else y2
        else:
            raise ValueError("This method only accepts ax_num = 0,1,2.")


        return x_comp, y_comp, N
        
    def plot_distil_binary(self, ax, ax_fixed):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
        
                # Set limits for ax
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        
       # Initialize numpy arrays
        y_r_array = np.zeros((self.x_array_equib[:, 0].size, 2))
        y_s_array = np.zeros((self.x_array_equib[:, 0].size, 2))

        for i, x1 in enumerate(self.x_array_equib):
            y_r_array[i] = self.rectifying_step_xtoy(x1)
            y_s_array[i] = self.stripping_step_xtoy(x1)
            
        #Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])

        op_color = 'green'
        intersection_counter = 0
        for i in range(len(y_r_array)-1, 0, -1): #Iterate backwards starting at top of rectifying curve
            if (abs((y_r_array[i,0]) - self.y_array_equib[i,0]) <= 0.001): #rectifying line intersects equib
                intersection_counter += 1
            if (abs((y_r_array[i,0]) - y_s_array[i,0]) <= 0.001): # operating lines interect
                if ((y_r_array[i,0] < self.y_array_equib[i,0]) & (intersection_counter > 1)):
                    op_color = 'black'
                if (y_r_array[i,0] >= self.y_array_equib[i,0]): #intersection occurs above equilibrium curve
                    op_color = 'red'  
                break 
               
        #Plot the rectifying line
        r_max_index = int(1000 * self.xD[0])
        for i in range(len(self.y_r_array)):
            if (self.y_r_array[i,0] < self.y_array_equib[i,0]): #set min index where the rectifying line intersects operating line
                r_min_index = i
                break 
        
        
        #Plot the stripping line
        s_min_index = int(1000 * self.xB[0])
        for i in range(len(self.y_s_array)): #set max index where stripping line intersects equib curve
            if (self.y_s_array[i,0] > self.y_array_equib[i,0]):
                s_max_index = i
                break 

        if (op_color == 'green'):
            for i in range(len(self.y_r_array)):
                if (self.y_r_array[i,0] < self.y_s_array[i,0]): #find index where operating lines intersect
                    op_intersect_index = i
                    break 
            ax.plot(self.x_array_equib[op_intersect_index:r_max_index, 0], self.y_r_array[op_intersect_index:r_max_index, 0], color = op_color)
            ax.plot(self.x_array_equib[s_min_index:op_intersect_index, 0], self.y_s_array[s_min_index:op_intersect_index, 0], color = op_color)  
            ax.plot(self.x_array_equib[op_intersect_index:s_max_index, 0], self.y_s_array[op_intersect_index:s_max_index, 0], color = op_color, alpha = 0.5)
            ax.plot(self.x_array_equib[r_min_index:op_intersect_index, 0], self.y_r_array[r_min_index:op_intersect_index, 0], color = op_color, alpha = 0.5)         
        else:
            ax.plot(self.x_array_equib[r_min_index:r_max_index, 0], self.y_r_array[r_min_index:r_max_index, 0], color = op_color)
            ax.plot(self.x_array_equib[s_min_index:s_max_index, 0], self.y_s_array[s_min_index:s_max_index, 0], color = op_color)
        
        
        #Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')

        x_r_0, y_r_0 = self.find_rect_fixedpoints_binary(n=30)
        x_s_0, y_s_0 = self.find_strip_fixedpoints_binary(n=30)
        
        x_stages, y_stages, N_2 = self.compute_equib_stages_binary(2, x_r_0 + x_s_0)
        if (op_color == 'green'):
            ax.plot(x_stages, y_stages, linestyle='--', color='black', alpha = 0.3)
        else:
            xr_stages, yr_stages, N_r = self.compute_equib_stages_binary(1, self.x_r_fixed)
            ax.plot(xr_stages, yr_stages, linestyle='--', color='black', alpha = 0.3)
            xs_stages, ys_stages, N_s = self.compute_equib_stages_binary(0, self.x_s_fixed)
            ax.plot(xs_stages, ys_stages, linestyle='--', color='black', alpha = 0.3)


        ax.scatter(x_r_0 + x_s_0, y_r_0 + y_s_0, s=50, c="red")

        x_rect = self.compute_equib_stages_binary(1, x_r_0 + x_s_0)[0]
        ax_fixed.scatter(x_rect, [0]*len(x_rect), marker='o', color='blue', facecolors='none', edgecolors='blue', linewidths = 0.75)
        x_strip = self.compute_equib_stages_binary(0, x_r_0 + x_s_0)[0]
        ax_fixed.scatter(x_strip, [0]*len(x_strip), marker = '^', color = 'blue', facecolors='none', edgecolors='blue', linewidths = 0.75)
        ax_fixed.text(0.5, -5, f"Number of Stages: {N_2}", ha='center', va='center', transform=ax_fixed.transAxes)
        ax_fixed.yaxis.set_ticks([])
        ax.set_aspect('equal', adjustable='box')

        ax_fixed.spines['top'].set_visible(False)
        ax_fixed.spines['right'].set_visible(False)
        ax_fixed.spines['bottom'].set_visible(False)
        ax_fixed.spines['left'].set_visible(False)
        ax_fixed.axhline(0, color='black')  # y=0 line
        ax_fixed.set_xlim([0,1])
        ax_fixed.yaxis.set_ticks([])
        ax_fixed.yaxis.set_ticklabels([])

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_title("Equilibrium and Operating Lines")
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=35)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)

        return ax, ax_fixed
