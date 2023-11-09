import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import random as rand
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from utils.rand_comp_gen import *

class VLEModel:
    def __init__(self, num_comp: int, P_sys: float, comp_names, partial_pressure_eqs, use_jacobian=False):
        self.num_comp = num_comp
        self.P_sys = P_sys
        self.comp_names = comp_names
        self.use_jacobian = use_jacobian
        self.partial_pressure_eqs = partial_pressure_eqs
        
    def get_activity_coefficient(self, x_array, Temp = None)->np.ndarray:
        """
        Computes the activity coefficient for each component in the the model.

        Raises:
            NotImplementedError: Not implemented for base class

        Returns:
            np.ndarray: activity coefficient of each component
        """
        raise NotImplementedError
    
    def get_vapor_pressure(self, Temp)->np.ndarray:
        """
        Compute the vapor pressure for each component in the model.

        Raises:
            NotImplementedError: Not implemented for base class

        Returns:
            np.ndarray: vapor pressure of each component
        """
        raise NotImplementedError
    
    def convert_x_to_y(self, x_array:np.ndarray, temp_guess = None)->np.ndarray:
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (np.ndarray): Liquid mole fraction of each component.
            temp_guess (float): inital temperature guess for fsolve

        Returns:
            solution (np.ndarray): The solution from the fsolve function, which includes the vapor mole fractions and the system temperature.
        """
        
        # Compute the boiling points for each component
        boiling_points = [eq.get_boiling_point(self.P_sys) for eq in self.partial_pressure_eqs]

        #Provides a random guess for temp if no temp_guess was provided as a parameter
        if temp_guess == None:
            temp_guess = rand.uniform(np.amax(boiling_points), np.amin(boiling_points))

        # Use fsolve to find the vapor mole fractions and system temperature that satisfy the equilibrium conditions
        ier = 0 #fsolve results, 1 for convergence, else nonconvergence
        runs = 0
        while True:
            runs += 1
            if runs % 10000 == 0:
                print("Current Run from convert_x_to_y:",runs)
            try:
                random_number = generate_point_system_random_sum_to_one(self.num_comp) #generate random composition as intial guess
                new_guess = np.append(random_number,temp_guess) #create initial guess for composition and temperature
               
                #use fsolve with jacobian if provided
                if self.use_jacobian: 
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy, new_guess, args=(x_array,), full_output=True, xtol=1e-12, fprime=self.jacobian_x_to_y)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
                else:
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy, new_guess, args=(x_array,), full_output=True, xtol=1e-12, fprime=None)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
            except:
                continue

    def convert_y_to_x(self, y_array:np.ndarray, temp_guess = None)->np.ndarray:
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (np.ndarray): Vapor mole fraction of each component.
            temp_guess (float): inital temperature guess for fsolve

        Returns:
            solution (np.ndarray): The solution from the fsolve function, which includes the liquid mole fractions and the system temperature.
        """
        
        # Compute the boiling points for each component
        boiling_points = [eq.get_boiling_point(self.P_sys) for eq in self.partial_pressure_eqs]
        
        #Provides a random guess for temp if no temp_guess was provided as a parameter
        if temp_guess == None:
            temp_guess = rand.uniform(np.amax(boiling_points), np.amin(boiling_points))
        
        #Parallel to convert_x_to_y, refer to comments above        
        ier = 0
        runs = 0
        while True:
            runs += 1 
            try:
                if runs % 10000 == 0:
                    print("Current Run from convert_y_to_x:",runs)
                random_number = generate_point_system_random_sum_to_one(self.num_comp)
                new_guess = np.append(random_number, temp_guess)
                
                if self.use_jacobian:
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy2, new_guess, args=(y_array,), full_output=True, xtol=1e-12, fprime=self.jacobian_y_to_x)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
                else:
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy2, new_guess, args=(y_array,), full_output=True, xtol=1e-12, fprime=None)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
            except:
                continue
        
    def compute_Txy(self, vars:np.ndarray, x_array:np.ndarray)->list:
        """
        Computes the system of equations for the T-x-y calculations for convert_x_to_y.

        This function is used as the input to the fsolve function to find the roots 
        of the system of equations, which represent the equilibrium conditions for 
        the vapor-liquid equilibrium calculations.

        Args:
            vars (np.ndarray): A 1D array containing the initial guess for the 
                vapor mole fractions and the system temperature.
            x_array (np.ndarray): A 1D array containing the liquid mole fractions.

        Returns:
            eqs (list): A list of the residuals of the equilibrium equations
        """
        
        # Extract the vapor mole fractions and temperature from the vars array
        y_array = vars[:-1]
        Temp = vars[-1]
        # Compute the left-hand side of the equilibrium equations
        lefths = x_array * self.get_activity_coefficient(x_array, Temp=Temp) * self.get_vapor_pressure(Temp)

        # Compute the right-hand side of the equilibrium equations
        righths = y_array * self.P_sys

        # Form the system of equations by subtracting the right-hand side from the left-hand side
        # Also include the normalization conditions for the mole fractions
        eqs = (lefths - righths).tolist() + [np.sum(y_array) - 1]

        return eqs
    
    def compute_Txy2(self, vars:np.ndarray, y_array:np.ndarray)->list:
        """
        Computes the system of equations for the T-x-y calculations for convert_y_to_x.

        This function is used as the input to the fsolve function to find the roots 
        of the system of equations, which represent the equilibrium conditions for 
        the vapor-liquid equilibrium calculations.

        Args:
            vars (np.ndarray): A 1D array containing the initial guess for the 
                liquid mole fractions and the system temperature.
            y_array (np.ndarray): A 1D array containing the vapor mole fractions.

        Returns:
            eqs (list):  A list of the residuals of the equilibrium equations.
        """
        
        # Extract the liquid mole fractions and temperature from the vars array
        x_array = vars[:-1]
        Temp = vars[-1]

        # Compute the left-hand side of the equilibrium equations
        lhs = x_array * self.get_activity_coefficient(x_array, Temp=Temp) * self.get_vapor_pressure(Temp)
        
        # Compute the right-hand side of the equilibrium equations
        rhs = y_array * self.P_sys

        # Form the system of equations by subtracting the right-hand side from the left-hand side
        # Also include the normalization conditions for the mole fractions
        eqs = (lhs - rhs).tolist() + [np.sum(x_array) - 1]

        return eqs


    def plot_binary_Txy(self, data_points:int, comp_index:int, ax):
        """
        Plots the T-x-y diagram for a binary mixture on the given ax object.

        Args:
            data_points (int): Number of data points to use in the plot.
            comp_index (int): Index of the component to plot.
            ax (matplotlib.axes._axes.Axes): The matplotlib axis object to plot on.

        Raises:
            ValueError: If the number of components is not 2.
        """
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary mixtures.")

        # Create an array of mole fractions for the first component
        x1_space = np.linspace(0, 1, data_points)

        # Create a 2D array of mole fractions for both components
        x_array = np.column_stack([x1_space, 1 - x1_space])

        # Initialize lists to store the vapor mole fractions and system temperatures
        y_array, t_evaluated = [], []

        # Compute the vapor mole fractions and system temperatures for each set of liquid mole fractions
        for x in x_array:
            solution = self.convert_x_to_y(x)[0]
            y_array.append(solution[:-1])
            t_evaluated.append(solution[-1])

        # Convert the list of vapor mole fractions to a 2D numpy array
        y_array = np.array(y_array)

        # Use the passed ax object for plotting
        ax.plot(x_array[:, comp_index], t_evaluated, label="Liquid phase")
        ax.plot(y_array[:, comp_index], t_evaluated, label="Vapor phase")
        ax.set_title("T-x-y Diagram for " + self.__class__.__name__)
        ax.set_xlabel(f"Mole fraction of component {comp_index + 1}")
        ax.set_ylabel("Temperature")
        ax.legend()

    
    def plot_ternary_txy(self, data_points:int, keep_zero:int):
        """
        Plots the surface plots for the ternary system, and also plots a T-x-y diagram
        with a specific component's composition set to 0.

        Args:
            data_points (int): Number of data points to use in the plot.
            comp_index (int): Index of the component to set to 0

        Raises:
            ValueError: If the number of components is not 3.
        """

        if self.num_comp != 3:
            raise ValueError("This method can only be used for ternary mixtures.")
    
        if keep_zero == 1:
            x1s, x2s = np.meshgrid(np.linspace(0, 1, data_points), 
                        np.linspace(0, 1, data_points))

            T = np.zeros((data_points, data_points))
            y1s, y2s = np.zeros((data_points, data_points)), np.zeros((data_points, data_points))

            for i in range(data_points):
                for j in range(data_points):
                    if x1s[i, j] + x2s[i, j] > 1:
                        T[i, j] = float('nan')
                        y1s[i, j] = float('nan')
                        y2s[i, j] = float('nan')
                        x1s[i, j] = float('nan')
                        x2s[i, j] = float('nan')
                    else:
                        solution = self.convert_x_to_y(np.array([x1s[i, j], x2s[i, j], 1 - x1s[i, j] - x2s[i, j]]))[0]
                        y1s[i, j] = solution[0]
                        y2s[i, j] = solution[1]
                        T[i, j] = solution[3]
            fig = plt.figure(figsize=(15, 5))
            ax = plt.subplot(121, projection='3d')
            ax.plot_surface(x1s, x2s, T)
            ax.plot_surface(y1s, y2s, T)

            ax.set_title('Surface Plot of Ternary System')

            ax.set_xlabel('x1, x2')
            ax.set_ylabel('y1, y2')
            ax.set_zlabel('T')

            ax = plt.subplot(122)
            ax.plot(x1s[0,:], T[0,:])
            ax.plot(y1s[0,:], T[0,:])

            ax.set_title('T-x-y cross section of Ternary plot: keeping x2 fixed at 0')

            ax.set_xlabel('x1, y1')
            ax.set_ylabel('T')
        elif keep_zero == 2:
            x1s, x3s = np.meshgrid(np.linspace(0, 1, 100), 
                       np.linspace(0, 1, 100))

            T = np.zeros((100, 100))
            y1s, y3s = np.zeros((100, 100)), np.zeros((100, 100))

            for i in range(100):
                for j in range(100):
                    if x1s[i, j] + x3s[i, j] > 1:
                        T[i, j] = float('nan')
                        y1s[i, j] = float('nan')
                        y3s[i, j] = float('nan')
                        x1s[i, j] = float('nan')
                        x3s[i, j] = float('nan')
                    else:
                        solution = self.convert_x_to_y(np.array([x1s[i, j], 1 - x1s[i, j] - x3s[i, j], x3s[i, j]]))
                        y1s[i, j] = solution[0]
                        y3s[i, j] = solution[2]
                        T[i, j] = solution[3]
            fig = plt.figure(figsize=(15, 5))
            ax = plt.subplot(121, projection='3d')
            ax.plot_surface(x1s, 1-x1s-x3s, T)
            ax.plot_surface(y1s, 1-y1s-y3s, T)

            ax.set_title('Surface Plot of Ternary system: X-to-Y')

            ax.set_xlabel('x1, x2')
            ax.set_ylabel('y1, y2')
            ax.set_zlabel('T')

            ax = plt.subplot(122)
            ax.plot(x1s[0,:], T[0,:])
            ax.plot(y1s[0,:], T[0,:])

            ax.set_title('T-x-y cross section of Ternary plot: keeping x3 fixed at 0')

            ax.set_xlabel('x1, y1')
            ax.set_ylabel('T')
        elif keep_zero == 0:
            x2s, x1s = np.meshgrid(np.linspace(0, 1, 100), 
                       np.linspace(0, 1, 100))

            T = np.zeros((100, 100))
            y2s, y1s = np.zeros((100, 100)), np.zeros((100, 100))

            for i in range(100):
                for j in range(100):
                    if x2s[i, j] + x1s[i, j] > 1:
                        T[i, j] = float('nan')
                        y2s[i, j] = float('nan')
                        y1s[i, j] = float('nan')
                        x2s[i, j] = float('nan')
                        x1s[i, j] = float('nan')
                    else:
                        solution = self.convert_x_to_y(np.array([x1s[i, j], x2s[i, j], 1 - x1s[i, j] - x2s[i, j]]))[0]
                        y1s[i, j] = solution[0]
                        y2s[i, j] = solution[1]
                        T[i, j] = solution[3]
            fig = plt.figure(figsize=(15, 5))
            ax = plt.subplot(121, projection='3d')
            ax.plot_surface(x1s, x2s, T)
            ax.plot_surface(y1s, y2s, T)

            ax.set_title('Surface Plot of Ternary system: X-to-Y')

            ax.set_xlabel('x1, x2')
            ax.set_ylabel('y1, y2')
            ax.set_zlabel('T')

            ax = plt.subplot(122)
            ax.plot(x2s[0,:], T[0,:])
            ax.plot(y2s[0,:], T[0,:])

            ax.set_title('T-x-y cross section of Ternary plot: keeping x1 fixed at 0')

            ax.set_xlabel('x2, y2')
            ax.set_ylabel('T')
            
        plt.show()

    def plot_yx_binary(self, data_points:int=100):
        """
        Plots the y-x diagram for a binary mixture.
        
        Args:
            data_points (int): Number of data points to use in the plot. Default is 100.
        """
        
        # Initialize a figure for plotting
        plt.figure(figsize=(10, 6))
        
        # Create an array of mole fractions for the first component
        x_space = np.linspace(0, 1, data_points)
        y_space = []
        
        # Compute the vapor mole fractions for each set of liquid mole fractions
        for x1 in x_space:
            # The mole fraction of the second component in the liquid phase is 1 - x1
            x2 = 1 - x1
            # Initialize a mole fraction array for all components
            x_array = np.array([x1, x2])
            # Solve for the vapor-liquid equilibrium
            y_array = self.convert_x_to_y(x_array)[0]
            # Append the vapor mole fraction for the first component
            y_space.append(y_array[0])

        # Plot y vs. x for the binary mixture
        plt.plot(x_space, y_space, label="Component 1")
        plt.xlabel("Liquid phase mole fraction (Component 1)")
        plt.ylabel("Vapor phase mole fraction (Component 1)")
        plt.title("y-x Diagram for Binary Mixture")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    # group of functions used for jacobian calculations
    def jacobian_x_to_y(self, uvec, xvec):
        gammas_ders = self.get_gamma_ders(np.concatenate((xvec, np.array([uvec[-1]]))), l=0) # l is dummy value
        gammas = self.get_activity_coefficient(xvec, uvec[-1])
        jac = np.empty((self.num_comp+1, self.num_comp+1))
        for i in range(self.num_comp):
            for j in range(self.num_comp+1):
                if j == self.num_comp:
                    jac[i,j] = xvec[i]*self.get_Psat_i(i, uvec[-1])*gammas_ders[i,-1] + xvec[i]*gammas[i]*self.get_dPsatdT_i(i, uvec[-1])
                elif i == j:
                    jac[i,j] = -self.get_Psys()
                else:
                    jac[i,j] = 0
        jac[self.num_comp, :] = np.concatenate((np.ones(self.num_comp), np.zeros(1)))
        return jac


    def jacobian_y_to_x(self, uvec, yvec):
        gammas_ders = self.get_gamma_ders(uvec, l=0) # l is dummy value
        gammas = self.get_activity_coefficient(uvec[:-1], uvec[-1])
        jac = np.empty((self.num_comp+1, self.num_comp+1))
        for i in range(self.num_comp):
            for j in range(self.num_comp+1):
                if j == self.num_comp:
                    jac[i,j] = uvec[i]*gammas_ders[i, -1]*self.get_Psat_i(i, uvec[-1]) + uvec[i]*gammas[i]*self.get_dPsatdT_i(i, uvec[-1])
                elif i == j:
                    jac[i,j] = gammas[i]*self.get_Psat_i(i, uvec[-1]) + uvec[i]*gammas_ders[i,j]*self.get_Psat_i(i, uvec[-1])
                else:
                    jac[i,j] = uvec[i]*gammas_ders[i,j]*self.get_Psat_i(i, uvec[-1])
        jac[self.num_comp, :] = np.concatenate((np.ones(self.num_comp), np.zeros(1)))
        return jac

    def get_Psat_i(self, i, T):
        return self.partial_pressure_eqs[i].get_partial_pressure(T)
    def get_dPsatdT_i(self, i, T):
        return self.partial_pressure_eqs[i].get_dPsatdT(T)
    def get_Psys(self):
        return self.P_sys
    def get_gamma_ders(self, uvec, l):
        raise NotImplementedError('Jacobian not available for this model')

        

        
        
   

                
            
        
        


        