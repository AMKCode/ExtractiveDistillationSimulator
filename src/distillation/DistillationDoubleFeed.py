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
from distillation.DistillationModel import DistillationModel

class DistillationModelDoubleFeed(DistillationModel):
    def __init__(self, thermo_model:VLEModel, Fr: float, zF: np.ndarray, xFL: np.ndarray, xFU: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, qL = 1, qU = 1) -> None:
        D_B = (zF[0]-xB[0])/(xD[0]-zF[0])
        FL_B = (xD[0]-xB[0])/(Fr*(xD[0]-xFU[0])+xD[0]-xFL[0])
        
        # assume reflux is given, boil_up is not given, and qU and qL are 1.
        # later can do an if-else to generalize like in DistillationModel
        # according to eqn 5.22
        boil_up = ((reflux+1)*D_B)+(FL_B*(Fr*(qU-1))+(qL-1))

        # let self.xF = zF; self.q = qL
        super().__init__(thermo_model,zF,xD,xB,reflux)
        self.xFU = xFU # composition of entrainer fluid entering in upper feed
        self.qU = qU # quality of upper feed
        self.Fr = Fr # feed ratio = (entrainer flow rate)/(feed flow rate)
        self.xFL = xFL
        self.boil_up = boil_up
        self.qL = qL
    
    
    def middle_step_y_to_x(self, y_m_j: np.ndarray):
        """
        Method to calculate y in the middle section of the distillation column from given y.

        Args:
            y_m_j (np.ndarray): Mole fraction of each component in the vapor phase in the middle section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        h = (self.Fr*(self.xD[0]-self.xB[0]))/(self.Fr*(self.xFU[0]-self.xB[0])+self.xF[0]-self.xB[0])
        # x = block1 * y_m_k + block2 (eq 5.21a)
        block1 = (self.reflux+1+((self.qU-1)*h))/(self.reflux+(self.qU*h))
        block2 = (h*self.xFU-self.xD)/(self.reflux+self.qU*h)
        
        return (block1*y_m_j)+block2

    def middle_step_x_to_y(self, x_m_j: np.ndarray): 
        """
        Method to calculate y in the middle section of the distillation column from given x.

        Args:
            x_m_j (np.ndarray): Mole fraction of each component in the liquid phase in the middle section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        h = (self.Fr*(self.xD[0]-self.xB[0]))/(self.Fr*(self.xFU[0]-self.xB[0])+self.xF[0]-self.xB[0])

        block1 = (self.reflux+1+((self.qU-1)*h))/(self.reflux+(self.qU*h))
        block2 = (h*self.xFU-self.xD)/(self.reflux+self.qU*h)
        
        return (x_m_j-block2)/block1

    def find_rect_fixedpoints(self, n):
        pass
    
    def find_strip_fixedpoints(self, n):
        pass
    
    def plot_distil_strip(self, ax, ax_fixed):
        pass
    def plot_distil_strip(self, ax, ax_fixed):
        pass

        
        
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
                print("counter rect:", counter)
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
            if counter == 100:
                print("counter strip:", counter)
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 0.0000000001:
                return np.array(x_comp), np.array(y_comp)
                
            x1 = x2
            y1 = y2
    
    def compute_middle_stages(self, start_point:int):
        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0
        #Need to start from a set position on the stripping section; call it 'start_point' number of 
        #stages into the stripping section.  Find the comp at that stage and use that as (x1, y1)

        #this is not done correctly though
        x_strip_comp = self.compute_stripping_stages()[0]
        x1 = x_strip_comp[start_point, :]
        y1 = self.middle_step_x_to_y(x1)

        #print(x1, y1) #debugging
        
        while True:
            x_comp.append(x1)
            y_comp.append(y1)
            counter += 1
            
            y2 = self.thermo_model.convert_x_to_y(x1)[0][:-1]
            x_comp.append(x1)
            y_comp.append(y2)
            
            x2 = self.middle_step_y_to_x(y2)
            if counter == 100:
                print("counter middle:", counter)
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 0.0000000001:
                return np.array(x_comp), np.array(y_comp)
            if (np.any(x2 < 0) or np.any(y2 < 0)):
                return np.array(x_comp), np.array(y_comp)
            #print(x2,  y2) #debugging
            x1 = x2
            y1 = y2
        

    def plot_rect_strip_comp(self, ax: axes, middle_start):
        middle_start = (2*middle_start - 1) #This is just to make the indexing work
        x_rect_comp = self.compute_rectifying_stages()[0]
        x_strip_comp = self.compute_stripping_stages()[0]
        x_middle_comp = self.compute_middle_stages(start_point = middle_start)[0]


        #Extract x1 and x2 from arrays
        x1_rect = x_rect_comp[:, 0]
        x2_rect = x_rect_comp[:, 1]
        x1_strip = x_strip_comp[:, 0]
        x2_strip = x_strip_comp[:, 1]
        x1_middle = x_middle_comp[:,0]
        x2_middle = x_middle_comp[:,1]
        
        # Plot the line connecting the points
        ax.plot(x1_rect, x2_rect, '-D', label='Rectifying Line', color = "red")  # '-o' means a line with circle markers at each data point
        ax.plot(x1_strip, x2_strip, '-s', label='Stripping Line', color = "blue")  # '-o' means a line with circle markers at each data point
        ax.plot(x1_middle, x2_middle, '-s', label='Middle Section', color = "black")  # '-o' means a line with circle markers at each data point

        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='x', color='orange', label='xF', s = 100)
        ax.scatter(self.xB[0], self.xB[1], marker='x', color='purple', label='xB', s = 100)
        ax.scatter(self.xD[0], self.xD[1], marker='x', color='green', label='xD', s = 100)
        ax.scatter(self.xFL[0], self.xFL[1], marker='x', color='black', label='xFL', s = 100)
        ax.scatter(self.xFU[0], self.xFU[1], marker='x', color='blue', label='xFU', s = 100)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend()
        
    def compute_equib_stages(self, ax_num, fixed_points = []):
        pass
        
    def plot_distil(self, ax, ax_fixed):
        pass
   