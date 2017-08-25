"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import sys
import unittest
import warnings

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.exceptions import ConvergenceWarning

from src.imc import (IMC, _fw, _fw_hess, _fw_prime,
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
        args_w = {
            key: (H, xh, yh, bh, 0.01, l1, W.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        args_cost = {
            key: (xh, yh, bh, 0.01, l1, H.size, H.shape, W.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        self.args = {'args_w': args_w, 'args_cost': args_cost}
        self.input_cost = np.concatenate([H.flatten(), W.flatten()])
        self.input_w = W.flatten()

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_cost(self):
        """The _cost function should return the regularized sum squared error."""  # noqa
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

    def test_cost_prime(self):
        """The _cost_prime function should return the gradient of the regularized sum squared error with respect to W and H."""  # noqa
        expected_0 = np.array(
            [1630440.01, 1945552.02, 2260664.03, 2575776.04, 3924900.05,
             4684792.06, 5444684.07, 6204576.08, 2847880.01, 3537800.02,
             4227720.03, 7083496.04, 8802920.05, 10522344.06]
        )
        actual_0 = _cost_prime(self.input_cost, *self.args['args_cost'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [1630440.005, 1945552.005, 2260664.005, 2575776.005, 3924900.005,
             4684792.005, 5444684.005, 6204576.005, 2847880.005, 3537800.005,
             4227720.005, 7083496.005, 8802920.005, 10522344.005]
        )
        actual_1 = _cost_prime(self.input_cost, *self.args['args_cost'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [1630440.0075, 1945552.0125, 2260664.0175, 2575776.0225,
             3924900.0275, 4684792.0325, 5444684.0375, 6204576.0425,
             2847880.0075, 3537800.0125, 4227720.0175, 7083496.0225,
             8802920.0275, 10522344.0325]
        )
        actual_2 = _cost_prime(self.input_cost, *self.args['args_cost'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_cost_hess(self):
        """The _cost_hess function should return the hessian of the regularized sum squared error with respect to W and H."""  # noqa
        expected_0 = np.array(
            [4158880., 5039936.01, 5920992.02, 6802048.03, 9999880.04,
             12112496.05, 14225112.06, 16337728.07, 23516080.08, 29703800.09,
             35891520.1, 58391536.11, 73709720.12, 89027904.13]
        )
        hess_p = np.arange(14)
        actual_0 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [4158880., 5039936., 5920992., 6802048., 9999880., 12112496.,
             14225112., 16337728., 23516080., 29703800., 35891520.,
             58391536., 73709720., 89027904.]
        )
        actual_1 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [4158880., 5039936.005, 5920992.01, 6802048.015, 9999880.02,
             12112496.025, 14225112.03, 16337728.035, 23516080.04,
             29703800.045, 35891520.05, 58391536.055, 73709720.06,
             89027904.065]
        )
        actual_2 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][0.5])
        np.testing.assert_array_equal(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw(self):
        """The _fw function should return the regularized sum squared error."""
        expected_0 = 37011644.476444982
        actual_0 = _fw(self.input_w, *self.args['args_w'][0])
        self.assertEqual(
            expected_0, actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = 37011644.840000004
        actual_1 = _fw(self.input_w, *self.args['args_w'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = 37011644.658222489
        actual_2 = _fw(self.input_w, *self.args['args_w'][0.5])
        self.assertEqual(
            expected_2, actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw_prime(self):
        """The _fw_prime function should return the gradient of the regularized sum squared error with respect to W."""  # noqa
        expected_0 = np.array(
            [3567200.01, 4257120.02, 4947040.03, 5636960.04, 8871520.05,
             10590944.06, 12310368.07, 14029792.08]
        )
        actual_0 = _fw_prime(self.input_w, *self.args['args_w'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [3567200.005, 4257120.005, 4947040.005, 5636960.005, 8871520.005,
             10590944.005, 12310368.005, 14029792.005]
        )
        actual_1 = _fw_prime(self.input_w, *self.args['args_w'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [3567200.0075, 4257120.0125, 4947040.0175, 5636960.0225,
             8871520.0275, 10590944.0325, 12310368.0375, 14029792.0425]
        )
        actual_2 = _fw_prime(self.input_w, *self.args['args_w'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw_hess(self):
        """The _fw_hess function should return the hessian of the regularized sum squared error with respect to W."""  # noqa
        expected_0 = np.array(
            [6831280., 8631800.01, 10432320.02, 16961776.03, 21418520.04,
             25875264.05]
        )
        hess_p = np.arange(6)
        actual_0 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [6831280., 8631800., 10432320., 16961776., 21418520., 25875264.]
        )
        actual_1 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [6831280., 8631800.005, 10432320.01, 16961776.015, 21418520.02,
             25875264.025]
        )
        actual_2 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0.5])
        np.testing.assert_array_equal(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )
