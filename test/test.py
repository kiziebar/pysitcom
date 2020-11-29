import unittest
import numpy as np

from pysitcom import Comet, Sitcom


class TestComet(unittest.TestCase):

    def test_criteriaFail(self):
        with self.assertRaises(TypeError):
            Comet(["Bad", 0.5, 1])

    def test_criteriaType(self):
        with self.assertRaises(TypeError):
            Comet((0, 0.5, 1))

    def test_lackOfCriteria(self):
        with self.assertRaises(ValueError):
            Comet([])

    def test_noCoInRateCo(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[2, 3]])
            tmp.rate_co()

    def test_preferenceMejFail(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[0, 1]])
            tmp.generate_co()
            tmp.rate_co('mej')

    def test_preferenceValueFail(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[0, 1]])
            tmp.generate_co()
            tmp.rate_co(value=8)

    def test_preferenceOptionFail(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[0, 1]])
            tmp.generate_co()
            tmp.rate_co('bad')

    def test_noCoInRate(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[2, 3]])
            tmp.rate(np.asarray([[0.2, 0.3]]))

    def test_noCoInChangePreferenceCo(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[2, 3]])
            tmp.change_co_preference([0.1, 0.2])

    def test_highPreferenceValue(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[2, 3]])
            tmp.generate_co()
            tmp.change_co_preference([0.1, 2])

    def test_lowPreferenceValue(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[2, 3]])
            tmp.generate_co()
            tmp.change_co_preference([0.1, -1])

    def test_alternativesLen(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[2, 3], [2, 3, 4]])
            tmp.generate_co()
            tmp.rate(np.array([[1, 2, 3]]))

    def test_alternativesValue(self):
        with self.assertRaises(ValueError):
            tmp = Comet([[1, 2], [2, 5, 8]])
            tmp.generate_co()
            tmp.rate(np.array([[4, 20], [2, -1]]))


class TestSitcom(unittest.TestCase):

    def test_alternativesValue(self):
        with self.assertRaises(ValueError):
            alternatives = np.array([[4, 20], [2, -1]])
            preference = [0.2, 0.4]
            criteria = [[1, 2], [9, 10]]
            Sitcom(alternatives, preference, criteria)

    def test_preferenceFail(self):
        with self.assertRaises(ValueError):
            alternatives = np.array([[1, 9], [2, 9]])
            preference = [1.2, 0.4]
            criteria = [[1, 2], [9, 10]]
            Sitcom(alternatives, preference, criteria)

    def test_criteriaFail(self):
        with self.assertRaises(TypeError):
            alternatives = np.array([[1, 9], [2, 9]])
            preference = [1.2, 0.4]
            criteria = [["False", 2], [9, 10]]
            Sitcom(alternatives, preference, criteria)
