import numpy as np

class AntoineEquation:
    """
    A class that represents the Antoine equation for calculating the saturation pressure of pure components.

    The Antoine equation is a semi-empirical correlation between vapor pressure and temperature for pure components.

    Args:
        A (float): Antoine equation parameter.
        B (float): Antoine equation parameter.
        C (float): Antoine equation parameter.

    Attributes:
        A (float): Antoine equation parameter.
        B (float): Antoine equation parameter.
        C (float): Antoine equation parameter.
    """

    def __init__(self, A: float, B: float, C: float):
        self.A = A
        self.B = B
        self.C = C

    def get_partial_pressure(self, Temp: np.ndarray):
        """
        Calculates the saturation pressure at a given temperature using the Antoine equation.

        Args:
            Temp (np.ndarray): The temperature(s) at which to calculate the saturation pressure.

        Returns:
            np.ndarray: The calculated saturation pressure(s).
        """
        return np.power((self.A - self.B/(Temp + self.C)), 10)

    def get_temperature(self, partial_pressure: np.ndarray):
        """
        Calculates the temperature at a given saturation pressure using the Antoine equation.

        Args:
            partial_pressure (np.ndarray): The saturation pressure(s) at which to calculate the temperature.

        Returns:
            np.ndarray: The calculated temperature(s).
        """
        return (self.B/(self.A - np.log10(partial_pressure))) - self.C
