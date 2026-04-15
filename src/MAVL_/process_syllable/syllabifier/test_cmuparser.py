import unittest
from .cmuparser3 import *


class TestDictionary(unittest.TestCase):
    cmu_dict = CMUDictionary()

    def test_aaronson_has_two(self):
        self.assertEqual(len(self.cmu_dict["AARONSON"]), 2)
        self.assertEqual(self.cmu_dict["AARONSON"][0], "EH1 R AH0 N S AH0 N")
        self.assertEqual(self.cmu_dict["AARONSON"][1], "AA1 R AH0 N S AH0 N")

    def test_lawfully_has_one(self):
        self.assertEqual(len(self.cmu_dict["LAWFULLY"]), 1)


if __name__ == "__main__":
    unittest.main()
