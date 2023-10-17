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
from utils.AntoineEquation import *
from distillation.DistillationModel import *
from distillation.DistillationDoubleFeed import *
   
class phase_portraits():
    def __init__(self, thermo_model:VLEModel, distil_model:DistillationModel = None):
        self.distil_model = distil_model
        self.thermo_model = thermo_model
    
    def plot_residue_curve(self, ax, t_span, data_points: int = 15, init_comps = None):
        def dxdt(t, x):
            try:
                return x - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
    def plot_strip_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        def dxdt(t, x):
            try:
                return self.distil_model.stripping_step_xtoy(x_s_j=x) - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
    def plot_rect_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        def dxdt(t, x):
            try:
                # return self.distil_model.rectifying_step_xtoy(x_r_j=x) - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
                return x - self.distil_model.rectifying_step_ytox(self.thermo_model.convert_x_to_y(x)[0][:-1])
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
    def plot_middle_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        if not isinstance(self.distil_model, DistillationModelDoubleFeed):
            raise TypeError("Invalid operation")
        def dxdt(t, x):
            try:
                return -self.distil_model.middle_step_x_to_y(x) + self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)  

    def int_plot_path(self, ax, initial, t_span, num_points, dxdt):
        x0 = np.array(initial)
        dt = (t_span[1] - t_span[0]) / num_points
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        x_vals = [x0]

        x = x0
        for i, t in enumerate(t_eval):
            x = rk4_step(t, x, dt, dxdt)
            
            if x is None or np.isinf(x).any() or np.isnan(x).any() or (x > 1).any() or (x < 0).any():
                print("Integration stopped due to overflow, NaN values, or out-of-bound values.")
                break

            x_vals.append(x)
            
            # Plotting logic
            if i % 7 == 0:
                if dt > 0:
                    dx = x_vals[-1][0] - x_vals[-2][0]
                    dy = x_vals[-1][1] - x_vals[-2][1]
                    ax.arrow(x_vals[-2][0], x_vals[-2][1], dx, dy, head_width=0.02, head_length=0.02, fc='k', ec='k')
                else:
                    dx = x_vals[-2][0] - x_vals[-1][0]
                    dy = x_vals[-2][1] - x_vals[-1][1]
                    ax.arrow(x_vals[-1][0], x_vals[-1][1], dx, dy, head_width=0.02, head_length=0.02, fc='k', ec='k')

        x_vals = np.array(x_vals)
        ax.plot(x_vals[:, 0], x_vals[:, 1], color='red')

def rk4_step(t, x, dt, dxdt):
    try:
        k1 = dt * dxdt(t, x)
        k2 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k1)
        k3 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k2)
        k4 = dt * dxdt(t + dt, x + k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    except OverflowError:
        print("Overflow occurred during integration.")
        return None
