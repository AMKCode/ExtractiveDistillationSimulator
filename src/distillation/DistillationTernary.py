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
from distillation.DistillationSingleFeed import DistillationModelSingleFeed

class DistillationModelTernary(DistillationModelSingleFeed):

    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        super().__init__(thermo_model,xF,xD,xB,reflux,boil_up,q)
            
        
    def plot_rect_comp(self, ax):

        x_rect_comp = self.compute_rectifying_stages()[0]
        
        #Extract x1 and x2 from arrays
        x1_rect = x_rect_comp[:, 0]
        x2_rect = x_rect_comp[:, 1]

        # Plot the line connecting the points
        ax.plot(x1_rect, x2_rect, '-D', label='Rectifying Line', color = "red")  # '-o' means a line with circle markers at each data point
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        ax.legend()
        
    def compute_rectifying_stages(self):

        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0
        
        x1 = self.xD
        y1 = self.rectifying_step_xtoy(x1)

        while True:

            x_comp.append(x1)
            y_comp.append(y1)
            counter += 1
            x2 = self.thermo_model.convert_y_to_x(y1)[0][:-1]
            
            y2 = self.rectifying_step_xtoy(x2)
            
            x_comp.append(x2)
            y_comp.append(y2)
            
            if counter == 100000:
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 0.0000001:
                return np.array(x_comp), np.array(y_comp)
                
            x1 = x2
            y1 = y2

    def compute_stripping_stages(self):

        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0
        
        x1 = self.xB
        y1 = self.stripping_step_xtoy(x1)

        while True:
            x_comp.append(x1)
            y_comp.append(y1)
            counter += 1
            
            y2 = self.thermo_model.convert_x_to_y(x1)[0][:-1]
            x_comp.append(x1)
            y_comp.append(y2)
            
            x2 = self.stripping_step_ytox(y2)
            if counter == 5000:
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 1.0e-10:
                return np.array(x_comp), np.array(y_comp)
                
            x1 = x2
            y1 = y2

    def plot_rect_strip_comp(self, ax: axes):

        x_rect_comp  = self.compute_rectifying_stages()[0]
        x_strip_comp = self.compute_stripping_stages()[0]
        
        #Extract x1 and x2 from arrays
        '''
        x1_rect = x_rect_comp[:-1, 0]
        x1_rect_final = x_rect_comp[-1, 0]
        x2_rect = x_rect_comp[:-1, 1]
        x2_rect_final = x_rect_comp[-1, 1]
        x1_strip = x_strip_comp[:-1, 0]
        x1_strip_final = x_strip_comp[-1, 0]
        x2_strip = x_strip_comp[:-1, 1]
        x2_strip_final = x_strip_comp[-1,1]
        '''
        
        # Plot the line connecting the points
        ax.plot(x_rect_comp[:-1, 0], x_rect_comp[:-1, 1], '-D', label='Rectifying Line', color = "red")  # '-D' means a line with diamond markers at each data point
        ax.plot( x_strip_comp[:-1, 0],  x_strip_comp[:-1, 1], '-s', label='Stripping Line', color = "blue")  # '-s' means a line with box markers at each data point
        ax.plot( x_rect_comp[-1, 0],  x_rect_comp[-1, 1], '*', label='Operating Section Terminus', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint
        ax.plot( x_strip_comp[-1, 0],  x_strip_comp[-1, 1], '*', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint

        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='X', color='orange', label='xF', s = 100)
        ax.scatter(self.xB[0], self.xB[1], marker='X', color='purple', label='xB', s = 100)
        ax.scatter(self.xD[0], self.xD[1], marker='X', color='green',  label='xD', s = 100)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend()


    def plot_strip_comp(self, ax: axes):

        x_strip_comp = self.compute_stripping_stages()[0]
                
        ax.plot( x_strip_comp[:-1, 0],  x_strip_comp[:-1, 1], '-s', label='Stripping Line', color = "blue")  # '-s' means a line with box markers at each data point
        ax.plot( x_strip_comp[-1, 0],  x_strip_comp[-1, 1], '*', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint

        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='X', color='orange', label='xF', s = 100)
        ax.scatter(self.xB[0], self.xB[1], marker='X', color='purple', label='xB', s = 100)
                
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend()


    def plot_rect_comp(self, ax: axes):

        x_rect_comp  = self.compute_rectifying_stages()[0]
                        
        # Plot the line connecting the points
        ax.plot(x_rect_comp[:-1, 0], x_rect_comp[:-1, 1], '-D', label='Rectifying Line', color = "red")  # '-D' means a line with diamond markers at each data point
        ax.plot( x_rect_comp[-1, 0],  x_rect_comp[-1, 1], '*', label='Operating Section Terminus', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint
        
        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='X', color='orange', label='xF', s = 100)
        ax.scatter(self.xD[0], self.xD[1], marker='X', color='green',  label='xD', s = 100)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend()

   
