import unittest

from .. import random_forest as rf

RF = rf.RandomForestModel()
a = RF.tfidf_rf_mod1()

class TestStringMethods(unittest.TestCase):
    def test_random_forest(self):
        self.assertTrue(a>0 and a<1)

if __name__ == '__main__':
    unittest.main()
