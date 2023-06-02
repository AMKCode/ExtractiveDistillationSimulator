import unittest
from ThermodynamicModel import RaoultsLawModel

class TestRaoultsLawModelBinary(unittest.TestCase):
    def setUp(self):
        self.component_vapor_pressure = [10.0, 100.0]  # Example vapor pressures for two components
        self.model = RaoultsLawModel(82)

    def test_convert_x_to_y(self):
        x = [0.2, 0.8]
        expected_y = [0.2 * 10/ 82, 0.8 * 100/82]
        calculated_y = self.model.convert_x_to_y(x, [10,100])
        for i in range(len(expected_y)):
            self.assertAlmostEqual(calculated_y[i], expected_y[i])

    def test_convert_y_to_x(self):
        y = [0.2, 0.8]
        expected_x = [0.4, 0.6]
        calculated_x = self.model.convert_y_to_x(y, [41,109+1/3])
        for i in range(len(expected_x)):
            self.assertAlmostEqual(calculated_x[i], expected_x[i])

if __name__ == '__main__':
    unittest.main()
