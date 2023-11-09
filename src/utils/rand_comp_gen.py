import numpy as np
import random as rand

def generate_point_system_random_sum_to_one(c:int):
    """
    Generates a random point in a C-component system such that the sum of all components equals 1.
    
    Parameters:
    c (int): Number of components in the system.
    
    Returns:
    np.ndarray: A numpy array containing the random point.
    """
    # Generate a seed using SystemRandom
    seed = int(rand.SystemRandom().random() * 1e9)
    
    # Seed NumPy's random number generator
    np.random.seed(seed)
    
    # Generate a point from a Dirichlet distribution with alpha = [1, 1, ..., 1] (c times)
    x = np.random.dirichlet([1]*c)
    
    # Ensure the sum is 1 by normalizing (Dirichlet should already output a sum-to-one array, but this is a good check)
    x_normalized = x / np.sum(x)
    
    return x_normalized

def create_restricted_simplex_grid(n_dimensions, resolution, max_value=0.999):
    """
    Create a grid of points for an n-dimensional simplex where sum(x_i) = 1
    and 0 < x_i < max_value for all i, excluding the extreme points from the start.

    Args:
    n_dimensions (int): The number of dimensions (or components) in the simplex.
    resolution (int): The number of intervals along each axis.
    max_value (float): The maximum value for any component in the simplex.

    Returns:
    list: List of tuples where each tuple represents a point in the simplex.
    """
    
    # Generate linearly spaced points between 0 and max_value
    points = np.linspace(0, max_value, resolution + 1)
    
    # Initialize list to hold the grid points
    grid_points = []
    
    # Generate all possible combinations for the first n-1 variables
    def generate_points(current_dimensions):
        if len(current_dimensions) == n_dimensions - 1:
            last_dimension = 1 - sum(current_dimensions)
            if 0 < last_dimension <= max_value:
                grid_points.append(tuple(current_dimensions + [last_dimension]))
        else:
            for point in points:
                new_dimensions = current_dimensions + [point]
                if sum(new_dimensions) < 1:
                    generate_points(new_dimensions)
    
    generate_points([])
    
    return grid_points

def create_simplex_grid(n_dimensions, resolution):
    """
    Create a grid of points for an n-dimensional simplex where sum(x_i) = 1
    and 0 < x_i < 1 for all i.

    Args:
    n_dimensions (int): The number of dimensions (or components) in the simplex.
    resolution (int): The number of intervals along each axis.

    Returns:
    list: List of tuples where each tuple represents a point in the simplex.
    """

    # Generate linearly spaced points between 0 and 1
    points = np.linspace(0, 1, resolution + 1)
    
    # Initialize list to hold the grid points
    grid_points = []
    
    # Generate all possible combinations for the first n-1 variables
    def generate_points(current_dimensions):
        if len(current_dimensions) == n_dimensions - 1:
            last_dimension = 1 - sum(current_dimensions)
            if last_dimension >= 0:
                grid_points.append(tuple(current_dimensions + [last_dimension]))
        else:
            for point in points:
                new_dimensions = current_dimensions + [point]
                if sum(new_dimensions) <= 1:
                    generate_points(new_dimensions)
    
    generate_points([])
    
    return grid_points


