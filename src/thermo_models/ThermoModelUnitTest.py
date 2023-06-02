import unittest
from ThermodynamicModel import RaoultsLawModel

class TestRaoultsLawModelBinary(unittest.TestCase):
    def setUp(self):
        self.component_vapor_pressure = [10.0, 100.0]  # Example vapor pressures for two components
        self.model = RaoultsLawModel()

    def test_convert_x_to_y(self):
        x = [0.2, 0.8]
        expected_y = [0.2 * 10/ 82, 0.8 * 100/82]
        calculated_y = self.model.convert_x_to_y(x)
        self.assertEqual(calculated_y, expected_y)

    def test_convert_y_to_x(self):
        y = [0.2, 0.8]
        expected_x = [1/1.4, 1 - 1/1.4]
        calculated_x = self.model.convert_y_to_x(y)
        self.assertEqual(calculated_x, expected_x)

if __name__ == '__main__':
    unittest.main()
