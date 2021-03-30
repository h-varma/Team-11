# author Johannes Fürle, 22.3.21
import unittest
import numpy as np
import johannes_circumference as cf


# python3 -m unittest test_johannes_circumference.py

class test_johannes_circumference(unittest.TestCase):
    def test_circumference(self):
        """Test different outcomes of the calculation of the circumference
        """
        self.assertEqual(cf.circumference_circle(1), 2 * np.pi)
        self.assertEqual(cf.circumference_circle(0), 0)
        self.assertEqual(cf.circumference_circle(10), 2 * np.pi * 10)

    def test_values(self):
        """check negative values
        """
        self.assertRaises(ValueError, cf.circumference_circle, -42)
