"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import sys
import unittest
import warnings

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.exceptions import ConvergenceWarning

from src.imc import (IMC, _fh, _fh_hess, _fh_prime,
                     _fit_inductive_matrix_completion, _cost, _cost_hess,
                     _cost_prime)


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
        self.imcs = {
            'imc0': IMC(max_iter=20, alpha=0.01, l1_ratio=0),
            'imck': IMC(n_components=2),
            'imcw': IMC(n_components=2, max_iter=2)
        }
        H = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        W = np.array([[1, 2, 3], [4, 5, 6]])
        R = np.array([[300, 6000], [9000, 12000]])
        rows, cols = lil_matrix(R).nonzero()
        xh = X[rows]
        yh = Y[rows]
        bh = R[rows, cols]
        self.data = {'R': R, 'X': X, 'Y': Y}
        args_h = {
            key: (W, xh, yh, bh, 0.01, l1, H.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        args_cost = {
            key: (xh, yh, bh, 0.01, l1, H.size, H.shape, W.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        self.args = {'args_h': args_h, 'args_cost': args_cost}
        self.input_cost = np.concatenate([H.flatten(), W.flatten()])
        self.input_h = H.flatten()

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_cost(self):
        """The _cost function should return the regularized sum squared error."""
        expected_0 = 37011644.476444982
        actual_0 = _cost(self.input_cost, *self.args['args_cost'][0])
        self.assertEqual(
            expected_0, actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = 37011644.840000004
        actual_1 = _cost(self.input_cost, *self.args['args_cost'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = 37011644.658222489
        actual_2 = _cost(self.input_cost, *self.args['args_cost'][0.5])
        self.assertEqual(
            expected_2, actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw_prime(self):
        """The _fw_prime function should return the gradient of the regularized sum squared error with respect to U."""  # noqa
        expected_0 = np.array(
            [809800.01, 1094120.02, 1378440.03, 2050600.04, 2758664.05,
             3466728.06]
        )
        actual_0 = _fw_prime(self.input_w, *self.args['args_w'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [809800.01, 1094120.01, 1378440.01, 2050600.01, 2758664.01,
             3466728.01]
        )
        actual_1 = _fw_prime(self.input_w, *self.args['args_w'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [809800.01, 1094120.015, 1378440.02, 2050600.025, 2758664.03,
             3466728.035]
        )
        actual_2 = _fw_prime(self.input_w, *self.args['args_w'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw_hess(self):
        """The _fw_hess function should return the hessian of the regularized sum squared error with respect to U."""  # noqa
        expected_0 = np.array(
            [4353520., 5755160.01, 7156800.02, 10723504.03, 14175992.04,
             17628480.05]
        )
        hess_p = np.arange(6)
        actual_0 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [4353520., 5755160., 7156800., 10723504., 14175992., 17628480.]
        )
        actual_1 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [4353520., 5755160.005, 7156800.01, 10723504.015, 14175992.02,
             17628480.025]
        )
        actual_2 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0.5])
        np.testing.assert_array_equal(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

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
        """Score method should return the root mean squared error for the reconstructed matrix."""  # noqa
        pass
