import math as math

class ThermodynamicModel:
    """
    Base class representing a thermodynamic model.

    This is an abstract base class that defines the interface for thermodynamic models.

    Methods:
        convert_x_to_y(x_array):
            Convert liquid mole fraction to vapor mole fraction.

        convert_y_to_x(y_array):
            Convert vapor mole fraction to liquid mole fraction.
    """
    def convert_x_to_y(self, x_array):
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (list[float]): Liquid mole fraction of each component.

        Raises:
            NotImplementedError: Base class method.
        """
        raise NotImplementedError("Method convert_x_to_y not implemented in base class")

    def convert_y_to_x(self, y_array):
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (list[float]): Vapor mole fraction of each component.

        Raises:
            NotImplementedError: Base class method.
        """
        raise NotImplementedError("Method convert_y_to_x not implemented in base class")