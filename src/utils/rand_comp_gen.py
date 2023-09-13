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

