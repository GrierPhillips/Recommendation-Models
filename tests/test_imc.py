"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import unittest

import numpy as np

from src.imc import IMC


class IMCTest(unittest.TestCase):
    """Suite of tests for the IMC class."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Teardown the test class."""
        pass

    def setUp(self):
        """Set up the IMC class before each test."""
        self.imc = IMC(rank=10, max_iter=20, lambda_=0.01)
        item_att_feats = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        user_atts = np.array([[1, 2, 3], [4, 5, 6]])
        item_atts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        user_att_feats = np.array([[1, 2, 3], [4, 5, 6]])
        ratings = np.array([[300, 6000], [9000, 12000]])
        self.args = (
            user_att_feats,
            user_atts,
            item_atts,
            ratings,
            self.imc.lambda_,
            item_att_feats.shape
        )
        self.in_array = item_att_feats.flatten()
        pass

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_fit(self):
        """The fit method should call the fit_transform method and return self."""  # noqa
        pass

    def test_fit_transform(self):
        """The fit_transform method should iterate through the algorithm until the fit condition is met, finally returning the W matrix."""  # noqa
        pass

    def test_fv(self):
        """The _fv function should return the regularized sum squared error."""
        expected = 26127448.952889957
        actual = self.imc._fv(self.in_array, *self.args)
        self.assertEqual(
            expected,
            actual,
            msg='Expected {}, but found {}.'.format(expected, actual)
        )
        pass

    def test_fv_prime(self):
        """The _fv_prime function should return the gradient of the regularized sum squared error with respect to V."""  # noqa
        expected = np.array(
            [568680.01, 641008.02, 713336.03, 785664.04, 1359780.05,
             1528912.06, 1698044.07, 1867176.08]
        )
        actual = self.imc._fv_prime(self.in_array, *self.args)
        np.testing.assert_array_equal(
            expected,
            actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual)
        )
        pass

    def test_fv_hess(self):
        """The _fv_hess function should return the hessian of the regularized sum squared error with respect to V."""  # noqa
        expected = np.array(
            [2622400., 3306176.01, 3989952.02, 4673728.03, 6261280.04,
             7893872.05, 9526464.06, 11159056.07]
        )
        hess_p = np.arange(8)
        actual = self.imc._fv_hess(self.in_array, hess_p, *self.args)
        np.testing.assert_array_equal(
            expected,
            actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual)
        )
        pass

    def test_fu(self):
        """The _fu function should return the regularized sum squared error."""
        expected = 26127448.952889957
        actual = self.imc._fu(self.in_array, *self.args)
        self.assertEqual(
            expected,
            actual,
            msg='Expected {}, but found {}.'.format(expected, actual)
        )
        pass

    def test_fu_prime(self):
        """The _fu_prime function should return the gradient of the regularized sum squared error with respect to U."""  # noqa
        expected = np.array(
            [809800.01, 1094120.02, 1378440.03, 2050600.04, 2758664.05,
             3466728.06]
        )
        actual = self.imc._fu_prime(self.in_array, *self.args)
        np.testing.assert_array_equal(
            expected,
            actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual)
        )
        pass

    def test_fu_hess(self):
        """The _fu_hess function should return the hessian of the regularized sum squared error with respect to U."""  # noqa
        expected = np.array(
            [4353520., 5755160.01, 7156800.02, 10723504.03, 14175992.04,
             17628480.05]
        )
        hess_p = np.arange(6)
        actual = self.imc._fu_hess(self.in_array, hess_p, *self.args)
        np.testing.assert_array_equal(
            expected,
            actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual)
        )
        pass

    def test_transform(self):
        """The transform method should return the W matrix from the fitted model."""  # noqa
        pass

    def test_predict_one(self):
        """The predict_one method should return the predicted rating for a given user, course pair."""  # noqa
        pass

    def test_predict_all(self):
        """The predict_all method should return the predicted ratings for all courses for a given user."""  # noqa
        pass

    def score(self):
        """The score method should return the root mean squared error for the reconstructed matrix."""  # noqa
        pass
