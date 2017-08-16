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
        self.imc0 = IMC(max_iter=20, alpha=0.01, l1_ratio=0)
        self.imc1 = IMC(max_iter=20, alpha=0.01, l1_ratio=1)
        self.imc0_5 = IMC(max_iter=20, alpha=0.01, l1_ratio=0.5)
        item_att_feats = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        user_atts = np.array([[1, 2, 3], [4, 5, 6]])
        item_atts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        user_att_feats = np.array([[1, 2, 3], [4, 5, 6]])
        ratings = np.array([[300, 6000], [9000, 12000]])
        self.args_v = {
            key: (
                user_att_feats, user_atts, item_atts, ratings, 0.01, l1,
                item_att_feats.shape
            ) for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        self.args_u = {
            key: (
                item_att_feats, user_atts, item_atts, ratings, 0.01, l1,
                user_att_feats.shape
            ) for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        self.input_v = item_att_feats.flatten()
        self.input_u = user_att_feats.flatten()

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
        expected_0 = 13063724.476444978
        actual_0 = self.imc0._fv(self.input_v, *self.args_v[0])
        self.assertEqual(
            expected_0,
            actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = 13063724.84
        actual_1 = self.imc1._fv(self.input_v, *self.args_v[1])
        self.assertEqual(
            expected_1,
            actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = 13063724.658222489
        actual_2 = self.imc0_5._fv(self.input_v, *self.args_v[0.5])
        self.assertEqual(
            expected_2,
            actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fv_prime(self):
        """The _fv_prime function should return the gradient of the regularized sum squared error with respect to V."""  # noqa
        pass

    def test_fv_hess(self):
        """The _fv_hess function should return the hessian of the regularized sum squared error with respect to V."""  # noqa
        pass

    def test_fu(self):
        """The _fu function should return the regularized sum squared error."""
        pass

    def test_fu_prime(self):
        """The _fu_prime function should return the gradient of the regularized sum squared error with respect to U."""  # noqa
        pass

    def test_fu_hess(self):
        """The _fu_hess function should return the hessian of the regularized sum squared error with respect to U."""  # noqa
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
