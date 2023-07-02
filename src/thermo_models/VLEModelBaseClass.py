import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import random as rand

class VLEModel:
    """
    A class used to represent a Vapor-Liquid Equilibrium (VLE) model.

    This base class provides methods for computing activity coefficients, vapor pressures, 
    and converting between liquid and vapor mole fractions. Derived classes provide
    specific calculation of activity coefficient and vapor pressure for different 
    types of mixtures.
    ...

    Attributes
    ----------
    num_comp : int
        The number of components in the mixture.
    P_sys : float
        The total pressure of the system.

    Methods
    -------
    get_activity_coefficient(*args) -> np.ndarray:
        Computes the activity coefficient for each component in the the model.
    get_vapor_pressure(*args) -> np.ndarray:
        Computes the vapor pressure for each component in the model.
    convert_x_to_y(x_array:np.ndarray) -> np.ndarray:
        Computes the conversion from liquid mole fraction to vapor mole fraction.
    compute_Txy(vars:np.ndarray, x_array:np.ndarray) -> list:
        Computes the system of equations for the T-x-y calculations.
    plot_binary_Txy(data_points:int, comp_index:int):
        Plots the T-x-y diagram for a binary mixture.
    """
    def __init__(self,num_comp:int,P_sys:float):
        self.num_comp = num_comp
        self.P_sys = P_sys
        
    def get_activity_coefficient(self, x_array=None)->np.ndarray:
        """
        Computes the activity coefficient for each component in the the model.

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """
        raise NotImplementedError
    
    def get_vapor_pressure(self, Temp)->np.ndarray:
        """
        Compute the vapor pressure for each component in the model.

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """
        raise NotImplementedError
    
    def convert_x_to_y(self, x_array:np.ndarray)->np.ndarray:
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (np.ndarray): Liquid mole fraction of each component.

        Returns:
            solution (np.ndarray): The solution from the fsolve function, which includes the vapor mole fractions and the system temperature.
        """
        # Compute the boiling points for each component
        boiling_points = [eq.get_boiling_point(self.P_sys) for eq in self.partial_pressure_eqs]

        # Estimate the system temperature as the average of the maximum and minimum boiling points
        Temp_guess = np.mean([np.amax(boiling_points), np.amin(boiling_points)])

        # Create an initial guess for the vapor mole fractions and system temperature
        init_guess = np.append(np.full(self.num_comp, 1/self.num_comp), Temp_guess)

        # Use fsolve to find the vapor mole fractions and system temperature that satisfy the equilibrium conditions
        solution, infodict, ier, mesg = fsolve(self.compute_Txy, init_guess, args=(x_array,), full_output=True)
        if ier != 0:
            for i in range(200):
                random_number = np.random.uniform(low = 0.0, high = 1.0, size = self.num_comp)
                new_guess = np.append(random_number/np.sum(random_number), rand.uniform(np.amax(boiling_points), np.amin(boiling_points)))
                solution, infodict, ier, mesg = fsolve(self.compute_Txy, new_guess, args=(x_array,), full_output=True)
                if ier == 1:
                    break
        return solution
    
    def convert_y_to_x(self, y_array:np.ndarray)->np.ndarray:
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (np.ndarray): Vapor mole fraction of each component.

        Returns:
            solution (np.ndarray): The solution from the fsolve function, which includes the liquid mole fractions and the system temperature.
        """
        # Compute the boiling points for each component
        boiling_points = [eq.get_boiling_point(self.P_sys) for eq in self.partial_pressure_eqs]
        
        # Estimate the system temperature as the average of the maximum and minimum boiling points
        Temp_guess = np.mean([np.amax(boiling_points), np.amin(boiling_points)])
        
         # Create an initial guess for the liquid mole fractions and system temperature
        init_guess = np.append(np.full(self.num_comp, 1/self.num_comp), Temp_guess)
        
        # Use fsolve to find the liquid mole fractions and system temperature that satisfy the equilibrium conditions
        solution, infodict, ier, mesg = fsolve(self.compute_Txy2, init_guess, args=(y_array,), full_output=True)
        if ier != 0:
            for i in range(200):
                random_number = np.random.uniform(low = 0.0, high = 1.0, size = self.num_comp)
                new_guess = np.append(random_number/np.sum(random_number), rand.uniform(np.amax(boiling_points), np.amin(boiling_points)))
                solution, infodict, ier, mesg = fsolve(self.compute_Txy2, new_guess, args=(y_array,), full_output=True)
                if ier == 1:
                    break
        return solution
        
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
            eqs (list): A list of equations representing the equilibrium conditions.
        """
        # Extract the vapor mole fractions and temperature from the vars array
        y_array = vars[:-1]
        Temp = vars[-1]

        # Compute the left-hand side of the equilibrium equations
        lefths = x_array * self.get_activity_coefficient(x_array) * self.get_vapor_pressure(Temp)

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
            eqs (list): A list of equations representing the equilibrium conditions.
        """
        
        # Extract the liquid mole fractions and temperature from the vars array
        x_array = vars[:-1]
        Temp = vars[-1]

        # Compute the left-hand side of the equilibrium equations
        lhs = x_array * self.get_activity_coefficient(x_array) * self.get_vapor_pressure(Temp)
        
        # Compute the right-hand side of the equilibrium equations
        rhs = y_array * self.P_sys

        # Form the system of equations by subtracting the right-hand side from the left-hand side
        # Also include the normalization conditions for the mole fractions
        eqs = (lhs - rhs).tolist() + [np.sum(x_array) - 1]

        return eqs

    def plot_binary_Txy(self, data_points:int, comp_index:int):
        """
        Plots the T-x-y diagram for a binary mixture.

        Args:
            data_points (int): Number of data points to use in the plot.
            comp_index (int): Index of the component to plot.

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
            solution = self.convert_x_to_y(x)
            y_array.append(solution[:-1])
            t_evaluated.append(solution[-1])

        # Convert the list of vapor mole fractions to a 2D numpy array
        y_array = np.array(y_array)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_array[:, comp_index], t_evaluated, label="Liquid phase")
        plt.plot(y_array[:, comp_index], t_evaluated, label="Vapor phase")
        plt.title("T-x-y Diagram")
        plt.xlabel(f"Mole fraction of component {comp_index + 1}")
        plt.ylabel("Temperature")
        plt.legend()
        plt.show()
    
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
                        solution = self.convert_x_to_y(np.array([x1s[i, j], x2s[i, j], 1 - x1s[i, j] - x2s[i, j]]))
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
                        solution = self.convert_x_to_y(np.array([x1s[i, j], x2s[i, j], 1 - x1s[i, j] - x2s[i, j]]))
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


        
   

                
            
        
        


        