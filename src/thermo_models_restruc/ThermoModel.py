import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib
import ternary

class VLEModel:
    def __init__(self,num_comp:int,P_sys:float):
        self.num_comp = num_comp
        self.P_sys = P_sys
    
    def convert_x_to_y(self, x_array:np.ndarray):
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (list[float]): Liquid mole fraction of each component.

        """ 
        temp, xandy = self.get_Txy(10)
        num_data_points = len(xandy)  # Number of data points
        x_solutions, y_solutions = np.split(np.array(xandy).reshape(num_data_points, self.num_comp*2), 2, axis=1)
        
        if self.num_comp == 2:
            self.plot_binary_Txy(x_solutions[:,0],y_solutions[:,0],temp)
        if self.num_comp == 3:
            self.plot_ternary_Tx(x_solutions[:,0],x_solutions[:,1],x_solutions[:,2])
            
    def convert_y_to_x(self, y_array):
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (list[float]): Vapor mole fraction of each component.

        Raises:
            NotImplementedError: Base class method.
        """
        raise NotImplementedError("Method convert_y_to_x not implemented in base class")
    
    def get_activity_coefficient(self,*args)->np.ndarray:
        raise NotImplementedError
    
    def get_vapor_pressure(self, *args)->np.ndarray:
        raise NotImplementedError
    
    def compute_Txy(self, vars, Temp):
        x_array = vars[:self.num_comp]
        y_array = vars[self.num_comp:]
        lefths = np.multiply(np.multiply(x_array, self.get_activity_coefficient(Temp)), self.get_vapor_pressure(Temp))
        righths = np.multiply(y_array, self.P_sys)
        eqs = (lefths - righths).tolist() + [np.sum(x_array) - 1] + [np.sum(y_array) - 1]
        return eqs

    def get_Txy(self, data_points):
        boiling_points = []
        for eq in self.partial_pressure_eqs:
            boiling_points.append(eq.get_boiling_point(self.P_sys))
        Temp_range = np.amax(boiling_points), np.amin(boiling_points)
        Temp_space = np.linspace(Temp_range[0],Temp_range[1],data_points)
        x_init = np.full(self.num_comp,1/self.num_comp)
        y_init = np.full(self.num_comp,1/self.num_comp)
        solutions = []
        for Temp in Temp_space:
            init_guess = np.concatenate([x_init, y_init])  # Flatten the initial guess into a 1D array
            solution = fsolve(self.compute_Txy, init_guess, args=(Temp,))
            solutions.append(solution)
        return Temp_space, solutions

    def plot_binary_Txy(self,x_array, y_array, t_evaluated):
        plt.figure(figsize=(10, 6))

        plt.plot(x_array, t_evaluated, label="Liquid phase")
        plt.plot(y_array, t_evaluated, label="Vapor phase")

        plt.title("T-x-y Diagram")
        plt.xlabel("Mole fraction of component 1")
        plt.ylabel("Temperature")
        plt.legend()
        plt.show()
        
    def plot_ternary_Tx(self,x_array_comp1, x_array_comp2, x_array_comp3):
        matplotlib.rcParams['figure.dpi'] = 200 # Make images higher resolution
        matplotlib.rcParams['figure.figsize'] = (4, 4) # and set default size
        # Sample trajectory plot
        figure, ax = ternary.figure(scale=100)

        # Define pure component names
        ax.right_corner_label("          Component 1", fontsize=8, offset=0.25)
        ax.top_corner_label("Component 2", fontsize=8, offset=0.2)
        ax.left_corner_label("Component 3", fontsize=8, offset=0.25)
        
        ax.boundary()
        ax.gridlines(multiple=10, color="black")
        ax.set_title("Ternary Diagram\n", fontsize=10)
        
        points = []
        for idx, x in x_array_comp1:
            points.append[x,x_array_comp2[idx],x_array_comp3[idx]]
        ax.plot(points=points)
        
        ax.get_axes().axis('off')
        ax.clear_matplotlib_ticks()
        ax.legend(loc='upper right', prop={'size': 5})
        ax.show()







                
            
        
        


        