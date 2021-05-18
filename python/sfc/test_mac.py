import mac_python_script
import unittest, math
import numpy as np

class TestMacPython(unittest.TestCase):

    def test_quadratic_scaling_factor_3d(self):
        '''Check the scaling factor against a simple calc and some known values.'''
        # 1) Define the input 
        h = 2
        # 2) Call the function 
        result = mac_python_script.quadratic_scaling_factor_3d(h)
        # 3) Verify the output
        self.assertEqual(result, 15.0 / (16.0 * math.pi * (h**3)))

    def test_quadratic(self):
        '''Check the quadratic function against an edge case.'''
        for test_h in np.linspace(0.001, 1.5E8, 100):
            self.assertEqual(mac_python_script.quadratic(h = test_h, r = 2*test_h), 0)
    
    def test_attractor_direction(self):
        '''Ensure the attractor function produces a force vector in the correct direction for various input.'''
        test_states = [
            ##(0, 0, 1, 1, 0, 0, 42, 960, 2),
            ##(0, 1, 1, 1, 1, 1, 1, 1, 1)
            {
                'x_pos':1,
                'y_pos':0,
                'z_pos':0,
            },
            {
                'x_pos':1,
                'y_pos':1,
                'z_pos':1,
            },
        ]
        test_params = {
            'x_target_pos':0,
            'y_target_pos':0,
            'z_target_pos':0,
            'h_attractor':2
        }
        F_0 = mac_python_script.F_attractor(0, test_states, test_params, {}).flatten()
        print(f'F_0: {F_0}')
        F_1 = mac_python_script.F_attractor(1, test_states, test_params, {}).flatten()
        print(f'F_1: {F_1}')
        self.assertGreater(0, F_0[0])
        self.assertEqual(0, F_0[1])
        self.assertEqual(0, F_0[2])
        self.assertGreater(0, F_1[0])
        self.assertGreater(0, F_1[0])
        self.assertGreater(0, F_1[0])
        self.assertGreater(0, F_1[0])
        self.assertEqual(F_1[0], F_1[1])
        self.assertEqual(F_1[2], F_1[1])
       
if __name__ == '__main__':
    unittest.main()



# max_i = 10
# epsilon = 0.001
# def exponential_func(i):
#     return i^2 + epsilon
# max_func_val = exponential_func(max_i)
# scaling_factor = 15e7 / max_func_val

# for i in range(max_i):
#     value = scaling_factor * exponential_func(i)
