import numpy as np

def check_overflow_np(value):
    if np.any(np.isinf(value)) or np.any(np.isnan(value)):
        raise ArithmeticError("Overflow encountered in calculation")