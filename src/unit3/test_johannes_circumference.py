# author Johannes Fürle, 22.3.21
import unittest
import numpy as np
import johannes_circumference as cf

# python3 -m unittest test_johannes_circumference.py

class test_johannes_circumference(unittest.TestCase):
    def test_circumference(self):
        self.assertEqual(cf.circumference_circle(1),2*np.pi)
