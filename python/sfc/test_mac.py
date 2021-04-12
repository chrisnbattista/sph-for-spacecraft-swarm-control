import mac_python_script
import unittest, math

class TestMacPython(unittest.TestCase):
    def test_quadratic_scaling_factor_3d(self):
        # 1) Define the input 
        h = 2
        # 2) Call the function 
        result = mac_python_script.quadratic_scaling_factor_3d(h)
        # 3) Verify the output
        self.assertEqual(result, 3.0/16/math.pi * h**3)

if __name__ == '__main__':
    unittest.main()





