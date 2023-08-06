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


#Notes:
#Conditions for a feasible column, profiles match at the feed stage  + no pinch point in between xB and xD
class DistillationModel:
    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        """
        DistillationModel constructor

        Args:
            thermo_model (VLEModel): Vapor-Liquid Equilibrium (VLE) model to be used in the distillation process.
            xF (np.ndarray): Mole fraction of each component in the feed.
            xD (np.ndarray): Mole fraction of each component in the distillate.
            xB (np.ndarray): Mole fraction of each component in the bottom product.
            reflux (Optional): Reflux ratio. If not provided, it will be calculated based on other parameters.
            boil_up (Optional): Boil-up ratio. If not provided, it will be calculated based on other parameters.
            q (float, optional): Feed condition (q) where q = 1 represents saturated liquid feed and q = 0 represents saturated vapor feed. Defaults to 1.
        
        Raises:
            ValueError: If the reflux, boil-up and q are not correctly specified. Only two of these parameters can be independently set.
        """
        self.thermo_model = thermo_model
        self.num_comp = thermo_model.num_comp
        self.xF = xF
        self.xD = xD
        self.xB = xB
        self.q = q
        
        if reflux is not None and boil_up is not None and q is None:
            self.boil_up = boil_up
            self.reflux = reflux
            self.q = ((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0])))-(reflux*((xF[0]-xB[0])/(xD[0]-xB[0]))) #this one need 1 component
        elif reflux is None and boil_up is not None and q is not None:
            self.boil_up = boil_up
            self.q = q
            self.reflux = (((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0]))) - self.q)/((xF[0]-xB[0])/(xD[0]-xB[0])) #this one need 1 component
        elif reflux is not None and boil_up is None and q is not None:
            self.reflux = reflux
            self.q = q
            self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1 #this one need 1 component
        else:
            raise ValueError("Underspecification or overspecification: only 2 variables between reflux, boil up, and q can be provided")
<<<<<<< HEAD
    
    def change_r(self, new_r):
        self.reflux = new_r
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
        return self
=======
        self.x_array_equib, self.y_array_equib, self.t_array = self.compute_equib() 
        
        # Initialize numpy arrays
        y_s_array = np.zeros((self.x_array_equib[:, 0].size, 2))
        y_r_array = np.zeros((self.x_array_equib[:, 0].size, 2))

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
        
>>>>>>> a7591fb295777a162a1b2d631d3bdbcca0a8121a
    def rectifying_step_xtoy(self, x_r_j:np.ndarray):
        """
        Method to calculate y in the rectifying section of the distillation column from given x.

        Args:
            x_r_j (np.ndarray): Mole fraction of each component in the liquid phase in the rectifying section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        r = self.reflux
        xD = self.xD
        return ((r/(r+1))*x_r_j)+((1/(r+1))*xD)

    def rectifying_step_ytox(self, y_r_j):
        """
        Method to calculate x in the rectifying section of the distillation column from given y.

        Args:
            y_r_j (float): Mole fraction of each component in the vapor phase in the rectifying section.

        Returns:
            float: Mole fraction of each component in the liquid phase in the rectifying section which corresponds to y_r_j.
        """
        r = self.reflux
        xD = self.xD
        return (((r+1)/r)*y_r_j - (xD/r))
    
    def stripping_step_ytox(self, y_s_j):
        """
        Method to calculate x in the stripping section of the distillation column from given y.

        Args:
            y_s_j (float): Mole fraction of each component in the vapor phase in the stripping section.

        Returns:
            float: Mole fraction of each component in the liquid phase in the stripping section that corresponds to y_s_j.
        """
        boil_up = self.boil_up
        xB = self.xB
        return ((boil_up/(boil_up+1))*y_s_j)+((1/(boil_up+1))*xB)
    
    def stripping_step_xtoy(self, x_s_j):
        """
        Method to calculate y in the stripping section of the distillation column from given x.

        Args:
            x_s_j (float): Mole fraction of each component in the liquid phase in the stripping section.

        Returns:
            float: Mole fraction of each component in the vapor phase in the stripping section.
        """
        boil_up = self.boil_up
        xB = self.xB
        return ((boil_up+1)/boil_up)*x_s_j - (xB/boil_up)
    
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
        return x0_values, y0_values 

    def compute_equib(self):
        x1_space = np.linspace(0, 1, 1000)
        y_array = np.zeros((x1_space.size, 2))
        t_array = np.zeros(x1_space.size)
        
        # Initialize numpy arrays
        x_array = np.zeros((x1_space.size, 2))
        for i, x1 in enumerate(x1_space):
            x_array[i] = [x1, 1 - x1]  # Fill the x_array directly
            solution = self.thermo_model.convert_x_to_y(x_array[i])[0]
            y_array[i] = solution[:-1]
            t_array[i] = solution[-1]
        return x_array, y_array, t_array
        
    def plot_distil_strip_binary(self, ax, ax_fixed):
        # Set limits for all main plots
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
            
        #Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])
        
        #Plot the stripping line
        ax.plot(self.x_array_equib[:, 0], self.y_s_array[:, 0], color = 'green')
        
        #Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')
        
        #Plot the fixed points from stripping line

        ax.scatter(self.x_s_fixed, self.y_s_fixed, s=50, c="red")
        x_fixed, y_fixed, N_1 = self.compute_equib_stages_binary(0, self.x_s_fixed)

        ax.plot(x_fixed, y_fixed, linestyle='--', color='black', alpha = 0.3)
        
        ax.scatter(self.x_s_fixed, self.y_s_fixed, s=50, c="red")

        ax_fixed.set_xlabel('$x_{1}$')
        ax_fixed.xaxis.set_label_coords(0.5, -0.05)
        ax_fixed.text(0.5, -5, f"Number of Stages: {N_1}", ha='center', va='center', transform=ax_fixed.transAxes)
        ax_fixed.yaxis.set_ticks([])

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

        return ax, ax_fixed

    def plot_distil_rect_binary(self, ax, ax_fixed):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
        
        # Set limits for ax
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

        #Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])
        
        #Plot the stripping line
        ax.plot(self.x_array_equib[:, 0], self.y_r_array[:, 0], color = 'green')
        
        #Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')

        
        x_stages, y_stages, N_2 = self.compute_equib_stages_binary(1, self.x_r_fixed)
        ax.plot(x_stages, y_stages, linestyle='--', color='black', alpha = 0.3)
        ax.scatter(self.x_r_fixed, self.y_r_fixed, s=50, c="red")

        ax_fixed.scatter(x_stages, [0]*len(x_stages), marker='x', color='green')
        ax_fixed.text(0.5, -5, f"Number of Stages: {N_2}", ha='center', va='center', transform=ax_fixed.transAxes)
        ax_fixed.yaxis.set_ticks([])
        ax.set_aspect('equal', adjustable='box')

        ax_fixed.scatter(self.x_r_fixed, [0]*len(self.x_r_fixed), marker='x', color='black')
        ax_fixed.spines['top'].set_visible(False)
        ax_fixed.spines['right'].set_visible(False)
        ax_fixed.spines['bottom'].set_visible(False)
        ax_fixed.spines['left'].set_visible(False)
        ax_fixed.axhline(0, color='black')  # y=0 line
        ax_fixed.set_xlim([0,1])
        ax_fixed.yaxis.set_ticks([])
        ax_fixed.yaxis.set_ticklabels([])

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_title("Equilibrium and Rectifying Line")

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
        
        #Plot the rectifying line
        ax.plot(self.x_array_equib[:, 0], y_r_array[:, 0], color = 'green')
        
        #Plot the stripping line
        ax.plot(self.x_array_equib[:, 0],y_s_array[:, 0], color = 'green' )
        
        #Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')

        x_r_0, y_r_0 = self.find_rect_fixedpoints_binary(n=30)
        x_s_0, y_s_0 = self.find_strip_fixedpoints_binary(n=30)
        
        x_stages, y_stages, N_2 = self.compute_equib_stages_binary(2, x_r_0 + x_s_0)
        
        ax.plot(x_stages, y_stages, linestyle='--', color='black', alpha = 0.3)
        ax.scatter(x_r_0 + x_s_0, y_r_0 + y_s_0, s=50, c="red")

        ax_fixed.scatter(x_stages, [0]*len(x_stages), marker='x', color='green')
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
        ax.set_title("Equilibrium and Rectifying Line")

        return ax, ax_fixed
        
    def plot_distil_binary_og(self, axs):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
            
        plots, fixed_plots = axs
        ax1, ax2, ax3 = plots
        ax1_fixed, ax2_fixed, ax3_fixed = fixed_plots
        
        # Set limits for all main plots
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])


        # Plotting main plots
        for ax in [ax1, ax2, ax3]:
            ax.plot(self.x_array_equib[:,0], self.y_array_equib[:,0])
            ax.plot([0,1], [0,1], linestyle='dashed')

        y_r = self.rectifying_step_xtoy(x1_space)
        y_s = self.stripping_step_xtoy(x1_space)
        

        '''
        Explanation of color changes:

        There are 3 cases represented by the colors.
        Case 1: Red - Operating Lines intersect above the equilibrium curve (Fixed Pt for Ex. 1)
        Case 2: Black - Operating Lines intersect beneath the equilibrium curve but a fixed point has
        been caused by a tangent pinch (as in Ex. 3)
        Case 3: Green - A feasible set of params.        
        ## There are other ways to represent the 3 cases than just color (linestyle, etc) ##
        
        This code iterates starting at the top of the rectifying line and moves toward the interectiom of operating lines
        ## Can the tangent pinch happen in the stripping section?  I would imagine so but I havent seen an example of this ##
        The code check for intersections between the rectofying line and equilibrium curve
        
        Next the code locates the intersection between operating lines
        If the operating lines intersect below the equilibrium curve:
            a. If the rectifying line has crossed the equilibrium curve before, the lines turn black
            b.  Otherwise the lines remain green
        If the operating lines intersect above the equilibrium curve, the lines turn red
        '''

        op_color = 'green'
        intersection_counter = 0
        for i in range(len(x1_space)-1, 0, -1): #Iterate backwards starting at top of rectifying curve
            if (abs((y_r[i]) - y_array[i,0]) <= 0.001): #rectifying line intersects equib
                intersection_counter += 1
            if (abs((y_r[i]) - y_s[i]) <= 0.001): # operating lines interect
                if ((y_r[i] < y_array[i,0]) & (intersection_counter > 1)):
                    op_color = 'black'
                if (y_r[i] >= y_array[i,0]): #intersection occurs above equilibrium curve
                    op_color = 'red'  
                break                        #Once the operating line intersect is found, the algorithm is complete
        #print(intersection_counter)


        '''
        #Currently causes infinite runtime!     
        x_equib = self.compute_equib_stages_binary(2)
        print(x_equib)
        '''


        ax3.plot(x1_space, y_r, color = op_color)
        ax2.plot(x1_space, y_r, color = op_color)
        ax3.plot(x1_space, y_s, color = op_color)
        ax1.plot(x1_space, y_s, color = op_color)
        
        x_r_0, y_r_0  = self.find_rect_fixedpoints_binary(n=30)
        x_s_0, y_s_0 = self.find_strip_fixedpoints_binary(n=30)
        
        #Compute the stage-wise composition
        x_ax1, y_ax1, N_1 = self.compute_equib_stages_binary(0, x_s_0)
        x_ax2, y_ax2, N_2 = self.compute_equib_stages_binary(1, x_r_0)
        x_ax3, y_ax3, N_3 = self.compute_equib_stages_binary(2, x_r_0 + x_s_0)
        
        #Plot the stage-wise composition on the second graphs
        ax1_fixed.scatter(x_ax1, [0]*len(x_ax1), marker='x', color='red')
        ax2_fixed.scatter(x_ax2, [0]*len(x_ax2), marker='x', color='green')
        
        #CHANGE THIS
        ax3_fixed.scatter(x_ax3, [0]*len(x_ax3), marker='x', color='red')

        
        
        ax1.plot(x_ax1, y_ax1, linestyle='--', color='black', alpha = 0.3)
        ax2.plot(x_ax2, y_ax2, linestyle='--', color='black', alpha = 0.3)
        ax3.plot(x_ax3, y_ax3, linestyle='--', color='black', alpha = 0.3)
        
        # Scatter plots
        ax3.scatter(x_r_0, y_r_0, s=50, c="red")
        ax2.scatter(x_r_0, y_r_0, s=50, c="red")
        ax3.scatter(x_s_0, y_s_0, s=50, c="red")
        ax1.scatter(x_s_0, y_s_0, s=50, c="red")

        # Set labels and titles for secondary plots
        ax1_fixed.set_xlabel('$x_{1}$')
        ax2_fixed.set_xlabel('$x_{1}$')
        ax3_fixed.set_xlabel('$x_{1}$')

        # Move the x1 label down for secondary plots (e.g., 0.02)
        ax1_fixed.xaxis.set_label_coords(0.5, -0.05)
        ax2_fixed.xaxis.set_label_coords(0.5, -0.05)
        ax3_fixed.xaxis.set_label_coords(0.5, -0.05)
        
        # Add descriptive text under the fixed graphs
        ax1_fixed.text(0.5, -5, f"Number of Stages: {N_1}", ha='center', va='center', transform=ax1_fixed.transAxes)
        ax2_fixed.text(0.5, -5, f"Number of Stages: {N_2}", ha='center', va='center', transform=ax2_fixed.transAxes)
        ax3_fixed.text(0.5, -5, f"Number of Stages: {N_3}", ha='center', va='center', transform=ax3_fixed.transAxes)

        # Disable y-axis for secondary plots
        ax1_fixed.yaxis.set_ticks([])
        ax2_fixed.yaxis.set_ticks([])
        ax3_fixed.yaxis.set_ticks([])
        

        # Set aspect for all plots
        for ax in [ax1, ax2, ax3]:
            ax.set_aspect('equal', adjustable='box')
        
        # Plot and format all fixed plots
        for i, (ax_fixed, x) in enumerate(zip([ax1_fixed, ax2_fixed, ax3_fixed], [x_s_0, x_r_0, x_r_0])):
            ax_fixed.scatter(x, [0]*len(x), marker='x', color='black')
            if i == 2:  # if we're dealing with the third plot
                ax_fixed.scatter(x_s_0, [0]*len(x_s_0), marker='x', color='black')
            ax_fixed.spines['top'].set_visible(False)
            ax_fixed.spines['right'].set_visible(False)
            ax_fixed.spines['bottom'].set_visible(False)
            ax_fixed.spines['left'].set_visible(False)
            ax_fixed.axhline(0, color='black')  # y=0 line
            ax_fixed.set_xlim([0,1])
            ax_fixed.yaxis.set_ticks([])
            ax_fixed.yaxis.set_ticklabels([])

            
        # Disable x-axis labels for the main plots
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        plt.setp(ax1.get_xticklabels(), visible=False)
        axs[0, 0].set_title("Equilibrium and Stripping Line")
        axs[0, 1].set_title("Equilibrium and Rectifying Line")
        axs[0, 2].set_title("Equilibrium and Operating Lines")
        
        return [ax1,ax2,ax3]

    def plot_distil_ternary(self):
        if self.num_comp != 3:
            raise ValueError("This method can only be used for binary distillation.")
