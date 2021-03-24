import unittest
import hridya_rectangle as hr


class TestHridyaRectangle(unittest.TestCase):
    def test_positivity(self):
        self.assertRaises(ValueError, hr.rectangle, -1, 0)
        self.assertRaises(ValueError, hr.rectangle, 2, -2)
        self.assertRaises(ValueError, hr.rectangle, -2, 2)

    def test_values(self):
        self.assertEqual(hr.rectangle(2, 3)["area"], 6)

    def test_commutativity(self):
        self.assertDictEqual(hr.rectangle(7, 3), hr.rectangle(3, 7))


if __name__ == '__main__':
    unittest.main()
