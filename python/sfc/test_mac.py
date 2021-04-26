import mac_python_script
import unittest, math
import numpy as np

class TestMacPython(unittest.TestCase):
    def test_quadratic_scaling_factor_3d(self):
        # 1) Define the input 
        h = 2
        # 2) Call the function 
        result = mac_python_script.quadratic_scaling_factor_3d(h)
        # 3) Verify the output
        self.assertEqual(result, 3.0/16/math.pi * h**3)
    def test_quadratic(self):
        for test_value in np.linspace(0.001, 150000000, 100):
            print(test_value)
            self.assertEqual(mac_python_script.quadratic(h = test_value, r = 2*test_value), 0)
       
if __name__ == '__main__':
    unittest.main()



max_i = 10
epsilon = 0.001
def exponential_func(i):
    return i^2 + epsilon
max_func_val = exponential_func(max_i)
scaling_factor = 15e7 / max_func_val

for i in range(max_i):
    value = scaling_factor * exponential_func(i)
