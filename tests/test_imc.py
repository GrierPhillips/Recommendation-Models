"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import sys
import unittest
import warnings

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.exceptions import ConvergenceWarning

from src.imc import (IMC, _check_init, _cost, _cost_hess, _cost_prime,
                     _fit_inductive_matrix_completion, _format_data, _fw,
                     _fw_hess, _fw_prime)


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
            'imc0': IMC(alpha=0.01, l1_ratio=0),
            'imck': IMC(n_components=2)}
        H = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        W = np.array([[1, 2, 3], [4, 5, 6]])
        R = np.array([[300, 6000], [9000, 12000]])
        rows, cols = lil_matrix(R).nonzero()
        xh = X[rows]
        yh = Y[rows]
        bh = R[rows, cols]
        self.data = {
            'r': bh, 'x': xh, 'y': yh, 'H': H, 'W': W, 'R': R, 'X': X, 'Y': Y}
        args_w = {
            key: (H, xh, yh, bh, 0.01, l1, W.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])}
        args_cost = {
            key: (xh, yh, bh, 0.01, l1, H.size, H.shape, W.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])}
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
            msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        expected_1 = 37011644.840000004
        actual_1 = _cost(self.input_cost, *self.args['args_cost'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = 37011644.658222489
        actual_2 = _cost(self.input_cost, *self.args['args_cost'][0.5])
        self.assertEqual(
            expected_2, actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_cost_prime(self):
        """The _cost_prime function should return the gradient of the regularized sum squared error with respect to W and H."""  # noqa
        expected_0 = np.array(
            [1630440.01, 1945552.02, 2260664.03, 2575776.04, 3924900.05,
             4684792.06, 5444684.07, 6204576.08, 2847880.01, 3537800.02,
             4227720.03, 7083496.04, 8802920.05, 10522344.06])
        actual_0 = _cost_prime(self.input_cost, *self.args['args_cost'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        expected_1 = np.array(
            [1630440.005, 1945552.005, 2260664.005, 2575776.005, 3924900.005,
             4684792.005, 5444684.005, 6204576.005, 2847880.005, 3537800.005,
             4227720.005, 7083496.005, 8802920.005, 10522344.005])
        actual_1 = _cost_prime(self.input_cost, *self.args['args_cost'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [1630440.0075, 1945552.0125, 2260664.0175, 2575776.0225,
             3924900.0275, 4684792.0325, 5444684.0375, 6204576.0425,
             2847880.0075, 3537800.0125, 4227720.0175, 7083496.0225,
             8802920.0275, 10522344.0325])
        actual_2 = _cost_prime(self.input_cost, *self.args['args_cost'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_cost_hess(self):
        """The _cost_hess function should return the hessian of the regularized sum squared error with respect to W and H."""  # noqa
        expected_0 = np.array(
            [4158880., 5039936.01, 5920992.02, 6802048.03, 9999880.04,
             12112496.05, 14225112.06, 16337728.07, 23516080.08, 29703800.09,
             35891520.1, 58391536.11, 73709720.12, 89027904.13])
        hess_p = np.arange(14)
        actual_0 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        expected_1 = np.array(
            [4158880., 5039936., 5920992., 6802048., 9999880., 12112496.,
             14225112., 16337728., 23516080., 29703800., 35891520.,
             58391536., 73709720., 89027904.])
        actual_1 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [4158880., 5039936.005, 5920992.01, 6802048.015, 9999880.02,
             12112496.025, 14225112.03, 16337728.035, 23516080.04,
             29703800.045, 35891520.05, 58391536.055, 73709720.06,
             89027904.065])
        actual_2 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][0.5])
        np.testing.assert_array_equal(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fw(self):
        """The _fw function should return the regularized sum squared error."""
        expected_0 = 37011644.476444982
        actual_0 = _fw(self.input_w, *self.args['args_w'][0])
        self.assertEqual(
            expected_0, actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        expected_1 = 37011644.840000004
        actual_1 = _fw(self.input_w, *self.args['args_w'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = 37011644.658222489
        actual_2 = _fw(self.input_w, *self.args['args_w'][0.5])
        self.assertEqual(
            expected_2, actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fw_prime(self):
        """The _fw_prime function should return the gradient of the regularized sum squared error with respect to W."""  # noqa
        expected_0 = np.array(
            [2847880.01, 3537800.02, 4227720.03, 7083496.04, 8802920.05,
             10522344.06])
        actual_0 = _fw_prime(self.input_w, *self.args['args_w'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        expected_1 = np.array(
            [2847880.005, 3537800.005, 4227720.005, 7083496.005, 8802920.005,
             10522344.005])
        actual_1 = _fw_prime(self.input_w, *self.args['args_w'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [2847880.0075, 3537800.0125, 4227720.0175, 7083496.0225,
             8802920.0275, 10522344.0325])
        actual_2 = _fw_prime(self.input_w, *self.args['args_w'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fw_hess(self):
        """The _fw_hess function should return the hessian of the regularized sum squared error with respect to W."""  # noqa
        expected_0 = np.array(
            [6831280., 8631800.01, 10432320.02, 16961776.03, 21418520.04,
             25875264.05])
        hess_p = np.arange(6)
        actual_0 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        expected_1 = np.array(
            [6831280., 8631800., 10432320., 16961776., 21418520., 25875264.])
        actual_1 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [6831280., 8631800.005, 10432320.01, 16961776.015, 21418520.02,
             25875264.025])
        actual_2 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0.5])
        np.testing.assert_array_equal(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fit_imc(self):
        """The _fit_inductive_matrix_completion function should solve for W and H or just W depending on the value of `update_H`, return W, H and a result message."""  # noqa
        expected_msg = (
            'Desired error not necessarily achieved due to precision loss.'
        )
        expected_w = np.array(
            [[-4.70285197, 1.67619005, 8.05523207],
             [-13.22177753, 1.09728147, 15.4163405]])
        _, _, succ, msg = _fit_inductive_matrix_completion(
            self.data['r'], self.data['x'], self.data['y'], verbose=1)
        actual_msg = sys.stdout.getvalue().split('\n')[0].split(': ')[1]
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        self.assertEqual(
            False, succ,
            msg='Expected {}, but found {}.'.format(False, succ))
        self.assertEqual(
            expected_msg, msg,
            msg='Expected {}, but found {}.'.format(expected_msg, msg))
        actual_w, _, succ, msg = _fit_inductive_matrix_completion(
            self.data['r'], self.data['x'], self.data['y'], H=self.data['H'],
            W=self.data['W'], n_components=2, update_H=False)
        np.testing.assert_allclose(
            actual_w, expected_w,
            err_msg='Expected {}, but found {}.'.format(expected_w, actual_w))

    def test_check_init(self):
        """The _check_init function should check to ensure that an array has a specified shape and is not all zeros, raising an error if not."""  # noqa
        with self.assertRaises(ValueError) as context:
            _check_init(self.data['H'], (3, 4), 'Check Init')
        expected_msgs = [
            'Array with wrong shape passed to Check Init. Expected (3, 4), ' +
            'but got (2, 4).',
            'Array passed to Check Init is full of zeros.']
        self.assertEqual(
            expected_msgs[0], str(context.exception),
            msg='Expected {}, but found {}.'.format(
                expected_msgs[0], context.exception))
        with self.assertRaises(ValueError) as context:
            _check_init(np.array([0, 0, 0]), (3, ), 'Check Init')
        self.assertEqual(
            expected_msgs[1], str(context.exception),
            msg='Expected {}, but found {}.'.format(
                expected_msgs[1], context.exception))

    def test_format_data(self):
        """The _format_data function should take in arrays for ratings, user attributes, and item attributes and return a 1-d array of ratings and 2-d arrays of attributes organized by the index of the rating."""  # noqa
        r, x, y = _format_data(self.data['R'], self.data['X'], self.data['Y'])
        self.assertEqual(
            r.shape, (4, ),
            msg='Expected {}, but found {}.'.format(r.shape, (4, )))
        self.assertEqual(
            x.shape, (4, 3),
            msg='Expected {}, but found {}.'.format(x.shape, (4, 3)))
        self.assertEqual(
            y.shape, (4, 4),
            msg='Expected {}, but found {}.'.format(y.shape, (4, 4)))

    def test_fit_transform(self):
        """The fit_transform method should call the _fit_inductive_matrix_completion method to fit the IMC, finally returning the W and H matrices and returning a warning if convergence is not achieved."""  # noqa
        expected_W = np.array(
            [[6.96299671, 0.07276383, -6.81746906],
             [-5.60313923, -2.55735318, 0.54]])
        expected_H = np.array(
            [[-7.3722333e+01, -4.8777391e+01, -2.3832449e+01, 1.11249223e+00],
             [-2.5343764e+00, -1.1122199e+00, 3.0993648e-01, 1.7320929e+00]])
        actual_W, actual_H = self.imcs['imc0']\
            .fit_transform(self.data['R'], self.data['X'], self.data['Y'])
        np.testing.assert_allclose(
            expected_W, actual_W[:2],
            err_msg='Expected {}, but found {}.'.format(expected_W, actual_W),
            atol=1e-1, rtol=1e-2
        )
        np.testing.assert_allclose(
            expected_H, actual_H[:2],
            err_msg='Expected {}, but found {}.'.format(expected_H, actual_H),
            atol=1e-2, rtol=1e-1
        )
        expected_W = np.array(
            [[6.96, 0.07, -6.81],
             [-5.13, -2.5, 0.14]])
        expected_H = np.array(
            [[-7.3722333e+01, -4.8777391e+01, -2.3832449e+01, 1.0],
             [-2.5343764e+00, -1.1122199e+00, 3.0993648e-01, 1.7320929e+00]])
        actual_W, actual_H = self.imcs['imck']\
            .fit_transform(self.data['R'], self.data['X'], self.data['Y'])
        np.testing.assert_allclose(
            expected_W, actual_W,
            err_msg='Expected {}, but found {}.'.format(expected_W, actual_W),
            rtol=1e-2, atol=1e-2
        )
        np.testing.assert_allclose(
            expected_H, actual_H,
            err_msg='Expected {}, but found {}.'.format(expected_H, actual_H),
            rtol=1e-2, atol=1e-1
        )
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.imcs['imck']\
                .fit_transform(self.data['R'], self.data['X'], self.data['Y'])
            self.assertEqual(warn[0].category, ConvergenceWarning)

    def test_fit(self):
        """The fit method should call the fit_transform method and return self."""  # noqa
        expected = True
        result = self.imcs['imc0']\
            .fit(self.data['R'], self.data['X'], self.data['Y'])
        actual_H = hasattr(result, 'components_h')
        actual_W = hasattr(result, 'components_w')
        self.assertEqual(
            expected, actual_W,
            msg='Expecte {}, but found {}.'.format(expected, actual_W)
        )
        self.assertEqual(
            expected, actual_H,
            msg='Expecte {}, but found {}.'.format(expected, actual_H)
        )

