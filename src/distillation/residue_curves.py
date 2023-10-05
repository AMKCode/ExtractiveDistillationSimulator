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
from matplotlib import axes
import random as rand
from utils.AntoineEquation import *
from thermo_models.RaoultsLawModel import *
from scipy.integrate import solve_ivp


class residue_curve():
    def __init__(self, thermo_model:VLEModel):
        self.thermo_model = thermo_model    
        
    def plot_residue_curve_og(self, ax, data_points: int = 15, init_comps = None):
        if init_comps ==  None:
            init_comps = []
            x1s, x2s = np.meshgrid(np.linspace(0, 1, data_points), 
                                    np.linspace(0, 1, data_points))

            for i in range(data_points):
                for j in range(data_points):
                    if x1s[i, j] + x2s[i, j] > 1 or (x1s[i,j])**2 + (x2s[i,j])**2 < 0.4 or x1s[i, j] * x2s[i, j] * (1 - (x1s[i, j] + x2s[i, j])) < 1e-6:
                        pass
                    else:
                        init_comps.append(np.array([x1s[i, j], x2s[i, j], 1 - x1s[i, j] - x2s[i, j]]))


        for init_comp in init_comps:
            def compute_residue_comp(init_comp):
                x_comp, y_comp = [], []  # Initialize composition lists
                counter = 0
                x1 = init_comp
                y1 = x1
                
                while True:
                    x_comp.append(x1)
                    y_comp.append(y1)
                    counter += 1
                    x2 = self.thermo_model.convert_y_to_x(y1)[0][:-1]
                    
                    y2 = x2
                    
                    x_comp.append(x2)
                    y_comp.append(y2)
                    
                    if counter == 100:
                        print("counter rect:", counter)
                        return np.array(x_comp), np.array(y_comp)
                    if np.linalg.norm(x1 - x2) < 0.0000001:
                        return np.array(x_comp), np.array(y_comp)
                        
                    x1 = x2
                    y1 = y2
            x_residue_com, _ = compute_residue_comp(init_comp=init_comp)
            #Extract x1 and x2 from arrays
            x1s = x_residue_com[:, 0]
            x2s = x_residue_com[:, 1]

            # Plot the line connecting the points
            ax.plot(x1s, x2s, '-', color = "red", linewidth = 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_title("Residue Curve - Acetone/Methanol/Water")
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
    def plot_residue_curve_mod(self, ax, data_points: int):
        init_comps = []
        x1s, x2s = np.meshgrid(np.linspace(0, 1, data_points), 
                                np.linspace(0, 1, data_points))

        for i in range(data_points):
            for j in range(data_points):
                if x1s[i, j] + x2s[i, j] > 1 or (x1s[i,j])**2 + (x2s[i,j])**2 < 0.40 or x1s[i, j] * x2s[i, j] * (1 - (x1s[i, j] + x2s[i, j])) < 1e-6:
                    pass
                else:
                    init_comps.append(np.array([x1s[i, j], x2s[i, j], 1 - x1s[i, j] - x2s[i, j]]))
        for init_comp in init_comps:
            def compute_residue_comp(init_comp):
                x_comp, y_comp = [], []  # Initialize composition lists
                counter = 0
                x1 = init_comp
                y1 = x1
                
                while True:
                    x_comp.append(x1)
                    y_comp.append(y1)
                    counter += 1
                    delta_x = (self.thermo_model.convert_y_to_x(y1)[0][:-1] - x1)*0.05
                    x2 = x1 + delta_x
                    delta_y = (x2 - y1)*0.05
                    y2 = y1 + delta_y
                    
                    x_comp.append(x2)
                    y_comp.append(y2)
                    
                    if counter == 1000000:
                        print("counter:", counter)
                        return np.array(x_comp), np.array(y_comp)
                    if np.linalg.norm(x1 - x2) < 0.0000001:
                        return np.array(x_comp), np.array(y_comp)
                        
                    x1 = x2
                    y1 = y2
            x_residue_com, _ = compute_residue_comp(init_comp=init_comp)
            
            #Extract x1 and x2 from arrays
            x1s = x_residue_com[:, 0]
            x2s = x_residue_com[:, 1]

            # Plot the line connecting the points
            ax.plot(x1s, x2s, '-', color = "red", linewidth = 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)


    def res_curve(self, ax, initial):
        def dxdt(t, x):
            try:
                return x - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None

        x0 = np.array(initial)
        t_span = [0, 10]
        num_points = 100
        dt = (t_span[1] - t_span[0]) / num_points
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        x_vals = [x0]

        x = x0
        for i, t in enumerate(t_eval):
            try:
                k1 = dt * dxdt(t, x)
                k2 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k1)
                k3 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k2)
                k4 = dt * dxdt(t + dt, x + k3)
                x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            except OverflowError:
                print("Overflow occurred during integration. Breaking loop.")
                break

            if np.isinf(x).any() or np.isnan(x).any():
                print("Integration stopped due to overflow or NaN values.")
                break

            x_vals.append(x)
            if i % 7 == 0:  # Plot an arrow every 5 points
                dx = x_vals[-1][0] - x_vals[-2][0]
                dy = x_vals[-1][1] - x_vals[-2][1]
                ax.arrow(x_vals[-2][0], x_vals[-2][1], dx, dy, head_width=0.02, head_length=0.02, fc='k', ec='k')

        x_vals = np.array(x_vals)
        ax.plot(x_vals[:, 0], x_vals[:, 1],color = 'black')
        
    def plot_residue_curve_int(self, ax, data_points: int = 15, init_comps = None):
        if init_comps ==  None:
            init_comps = []
            x1s, x2s = np.meshgrid(np.linspace(0, 1, data_points), 
                                    np.linspace(0, 1, data_points))

            for i in range(data_points):
                for j in range(data_points):
                    if x1s[i, j] + x2s[i, j] > 1 or (x1s[i,j])**2 + (x2s[i,j])**2 < 0.4 or x1s[i, j] * x2s[i, j] * (1 - (x1s[i, j] + x2s[i, j])) < 1e-6:
                        pass
                    else:
                        init_comps.append(np.array([x1s[i, j], x2s[i, j], 1 - x1s[i, j] - x2s[i, j]]))

        for init_comp in init_comps:
            self.res_curve(ax,init_comp)
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
    