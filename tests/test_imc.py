"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import sys
import unittest
import warnings

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from src.imc import (IMC, _check_init, _cost, _cost_hess, _cost_prime,
                     _fit_imc, _format_data, _fw,
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
        h_component = np.arange(1, 41).reshape(2, 20)
        user_atts = np.array([
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
        item_atts = np.array([
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
        w_component = np.arange(1, 31).reshape(2, 15)
        ratings_mat = np.array([[300, 6000], [9000, 12000]])
        rows, cols = csc_matrix(ratings_mat).nonzero()
        x_h = user_atts[rows]
        y_h = item_atts[rows]
        b_h = ratings_mat[rows, cols]
        self.data = {
            'r': b_h, 'x': x_h, 'y': y_h, 'H': h_component, 'W': w_component,
            'R': ratings_mat, 'X': user_atts, 'Y': item_atts}
        args_w = {
            key: (h_component, x_h, y_h, b_h, 0.01, l1, w_component.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])}
        args_cost = {
            key: (
                x_h, y_h, b_h, 0.01, l1, h_component.size,
                h_component.shape, w_component.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])}
        args_sparse = (
            csc_matrix(x_h), csc_matrix(y_h), b_h, 0.01, 0, h_component.size,
            h_component.shape, w_component.shape)
        args_ws = (
            h_component, csc_matrix(x_h), csc_matrix(y_h), b_h, 0.01, 0,
            w_component.shape)
        self.args = {
            'args_w': args_w, 'args_cost': args_cost, 'args_s': args_sparse,
            'args_ws': args_ws}
        self.input_cost = np.concatenate(
            [h_component.flatten(), w_component.flatten()])
        self.input_w = w_component.flatten()

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_cost(self):
        """The _cost function should return the regularized sum squared error."""  # noqa
        expected_0 = 41891260.920639724
        actual_0 = _cost(self.input_cost, *self.args['args_cost'][0])
        self.assertEqual(
            expected_0, actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        actual_s = _cost(self.input_cost, *self.args['args_s'])
        self.assertEqual(
            expected_0, actual_s,
            msg='Expected {}, but found {}.'.format(expected_0, actual_s))
        expected_1 = 41891260.199999988
        actual_1 = _cost(self.input_cost, *self.args['args_cost'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = 41891260.560319841
        actual_2 = _cost(self.input_cost, *self.args['args_cost'][0.5])
        self.assertEqual(
            expected_2, actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_cost_prime(self):
        """The _cost_prime function should return the gradient of the regularized sum squared error with respect to W and H."""  # noqa
        expected_0 = np.array(
            [235620.01, 0.02, 0.03, 0.04, -12095.95, 0.06, 235620.07, 0.08,
             -12095.91, 0.1, 0.11, 0.12, 223524.13, 0.14, 0.15, 0.16,
             -12095.83, 0.18, 235620.19, 0.2, 740520.21, 0.22, 0.23, 0.24,
             -31535.75, 0.26, 740520.27, 0.28, -31535.71, 0.3, 0.31, 0.32,
             708984.33, 0.34, 0.35, 0.36, -31535.63, 0.38, 740520.39, 0.4,
             448800.01, 0.02, 0.03, -19007.96, 0.05, 0.06, 448800.07, 0.08,
             0.09, -19007.9, 0.11, 0.12, 448800.13, -19007.86, 0.15,
             1346400.16, 0.17, 0.18, -53567.81, 0.2, 0.21, 1346400.22, 0.23,
             0.24, -53567.75, 0.26, 0.27, 1346400.28, -53567.71, 0.3])
        actual_0 = _cost_prime(self.input_cost, *self.args['args_cost'][0])
        np.testing.assert_allclose(
            expected_0, actual_0, rtol=1e-1, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        actual_s = _cost_prime(self.input_cost, *self.args['args_s'])
        np.testing.assert_allclose(
            expected_0, actual_s, rtol=1e-1, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_s))
        expected_1 = np.array(
            [235620.005, 0.005, 0.005, 0.005, -12095.995, 0.005, 235620.005,
             0.005, -12095.995, 0.005, 0.005, 0.005, 223524.005, 0.005, 0.005,
             0.005, -12095.995, 0.005, 235620.005, 0.005, 740520.005, 0.005,
             0.005, 0.005, -31535.995, 0.005, 740520.005, 0.005, -31535.995,
             0.005, 0.005, 0.005, 708984.005, 0.005, 0.005, 0.005, -31535.995,
             0.005, 740520.005, 0.005, 448800.005, 0.005, 0.005, -19007.995,
             0.005, 0.005, 448800.005, 0.005, 0.005, -19007.995, 0.005, 0.005,
             448800.005, -19007.995, 0.005, 1346400, 0.005, 0.005, -53567.995,
             0.005, 0.005, 1346400, 0.005, 0.005, -53567.995, 0.005, 0.005,
             1346400, -53567.995, 0.005])
        actual_1 = _cost_prime(self.input_cost, *self.args['args_cost'][1])
        np.testing.assert_allclose(
            expected_1, actual_1, rtol=1e-1, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [235620.008, 0.0125, 0.0175, 0.0225, -12095.9725, 0.0325,
             235620.038, 0.0425, -12095.9525, 0.0525, 0.0575, 0.0625,
             223524.068, 0.0725, 0.0775, 0.0825, -12095.9125, 0.0925,
             235620.098, 0.1025, 740520.107, 0.1125, 0.1175, 0.1225,
             -31535.8725, 0.1325, 740520.137, 0.1425, -31535.8525, 0.1525,
             0.1575, 0.1625, 708984.167, 0.1725, 0.1775, 0.1825, -31535.8125,
             0.1925, 740520.197, 0.2025, 448800.008, 0.0125, 0.0175,
             -19007.9775, 0.0275, 0.0325, 448800.037, 0.0425, 0.0475,
             -19007.9475, 0.0575, 0.0625, 4.48800068e+05, -19007.9275, 0.0775,
             1346400.08, 0.0875, 0.0925, -53567.9025, .0125, 0.1075,
             1346400.11, 0.1175, 0.1225, -53567.8725, 0.1325, 0.1375,
             1346400.14, -53567.8525, 0.1525])
        actual_2 = _cost_prime(self.input_cost, *self.args['args_cost'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2, rtol=1e-2, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_cost_hess(self):
        """The _cost_hess function should return the hessian of the regularized sum squared error with respect to W and H."""  # noqa
        expected_0 = np.array(
            [353304, 0.01, 0.02, 0.03, 553280.04, 0.05, 353304.06, 0.07,
             553280.08, 0.09, 0.1, 0.11, 906584.12, 0.13, 0.14, 0.15,
             553280.16, 0.17, 353304.18, 0.19, 1110384.20, 0.21, 0.22, 0.23,
             1442480.24, 0.25, 1110384.26, 0.27, 1442480.28, 0.29, 0.3, 0.31,
             2552864.32, 0.33, 0.34, 0.35, 1442480.36, 0.37, 1110384.38, 0.39,
             2198400.4, 0.41, 0.42, 2634720.43, 0.44, 0.45, 2198400.46, 0.47,
             0.48, 2634720.49, 0.5, 0.51, 2198400.52, 2634720.53, 0.54,
             6595200.55, 0.56, 0.57, 7425120.58, 0.59, 0.6, 6595200.61, 0.62,
             0.63, 7425120.64, 0.65, 0.66, 6595200.67, 7425120.68, 0.69])
        hess_p = np.arange(70)
        actual_0 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][0])
        np.testing.assert_allclose(
            expected_0, actual_0, rtol=1e-2, atol=1e-2,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        actual_s = _cost_hess(self.input_cost, hess_p, *self.args['args_s'])
        np.testing.assert_allclose(
            expected_0, actual_s, rtol=1e-2, atol=1e-2,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_s))
        expected_1 = np.array(
            [353304, 0, 0, 0, 553280, 0, 353304, 0, 553280, 0, 0, 0, 906584, 0,
             0, 0, 553280, 0, 353304, 0, 1110384, 0, 0, 0, 1442480, 0, 1110384,
             0, 1442480, 0, 0, 0, 2552864, 0, 0, 0, 1442480, 0, 1110384, 0,
             2198400, 0, 0, 2634720, 0, 0, 2198400, 0, 0, 2634720, 0, 0,
             2198400, 2634720, 0, 6595200, 0, 0, 7425120, 0, 0, 6595200, 0, 0,
             7425120, 0, 0, 6595200, 7425120, 0])
        actual_1 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][1])
        np.testing.assert_allclose(
            expected_1, actual_1, rtol=1e-2, atol=1e-2,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [353304, 0.005, 0.01, 0.015, 553280.02, 0.025, 353304.03, 0.035,
             553280.04, 0.045, 0.05, 0.055, 906584.06, 0.065, 0.07, 0.075,
             553280.08, 0.085, 353304.09, 0.095, 1110384.1, 0.105, 0.11, 0.115,
             1442480.12, 0.125, 1110384.13, 0.135, 1442480.14, 0.145, 0.15,
             0.155, 2552864.16, 0.165, 0.17, 0.175, 1442480.18, 0.185,
             1110384.19, 0.195, 2198400.2, 0.205, 0.21, 2634720.21, 0.22,
             0.225, 2198400.23, 0.235, 0.24, 2634720.25, 0.25, 0.255,
             2198400.26, 2634720.27, 0.27, 6595200.28, 0.28, 0.285, 7425120.29,
             0.295, 0.3, 6595200.3, 0.31, 0.315, 7425120.32, 0.325, 0.33,
             6595200.33, 7425120.34, 0.345])
        actual_2 = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2, rtol=1e-2, atol=1e-2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fw(self):
        """The _fw function should return the regularized sum squared error."""
        expected_0 = 41891260.920639724
        actual_0 = _fw(self.input_w, *self.args['args_w'][0])
        self.assertEqual(
            expected_0, actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        actual_s = _fw(self.input_w, *self.args['args_ws'])
        self.assertEqual(
            expected_0, actual_s,
            msg='Expected {}, but found {}.'.format(expected_0, actual_s))
        expected_1 = 41891260.199999988
        actual_1 = _fw(self.input_w, *self.args['args_w'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = 41891260.560319841
        actual_2 = _fw(self.input_w, *self.args['args_w'][0.5])
        self.assertEqual(
            expected_2, actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fw_prime(self):
        """The _fw_prime function should return the gradient of the regularized sum squared error with respect to W."""  # noqa
        expected_0 = np.array(
            [4.488e+05, 0.02, 0.03, -1.9007e+04, 0.05, 0.06, 4.488e+05, 0.08,
             0.09, -1.9007e+04, 0.11e-01, 0.12, 4.488e+05, -1.9007e+04, 0.15,
             1.3464e+06, 0.17, 0.18, -5.3567e+04, 0.2, 0.21, 1.3464e+06, 0.23,
             0.24, -5.3567e+04, 0.26, 0.27, 1.3464e+06, -5.3567e+04, 0.3])
        actual_0 = _fw_prime(self.input_w, *self.args['args_w'][0])
        np.testing.assert_allclose(
            expected_0, actual_0, rtol=1e-1, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        actual_s = _fw_prime(self.input_w, *self.args['args_ws'])
        np.testing.assert_allclose(
            expected_0, actual_s, rtol=1e-1, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_s))
        expected_1 = np.array(
            [4.488e+05, 0.005, 0.005, -1.9007e+04, 0.005, 0.005, 4.488e+05,
             0.005, 0.005, -1.9007e+04, 0.005, 0.005, 4.488e+05, -1.9007e+04,
             0.005, 1.3464e+06, 0.005, 0.005, -5.3567e+04, 0.005, 0.005,
             1.3464e+06, 0.005, 0.005, -5.3567e+04, 0.005, 0.005, 1.3464e+06,
             -5.3567e+04, 0.005])
        actual_1 = _fw_prime(self.input_w, *self.args['args_w'][1])
        np.testing.assert_allclose(
            expected_1, actual_1, rtol=1e-1, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [4.488e+05, 0.0125, 0.0175, -1.9007e+04, 0.0275, 0.0325, 4.488e+05,
             0.0425, 0.0475, -1.9007e+04, 0.0575, 0.0625, 4.488e+05,
             -1.9007e+04, 0.0775, 1.3464e+06, 0.0875, 0.0925, -5.3567e+04,
             0.1025, 0.1075, 1.3464e+06, 0.1175, 0.1225, -5.3567e+04, 0.1325,
             0.1375, 1.3464e+06, -5.3567e+04, 0.1525])
        actual_2 = _fw_prime(self.input_w, *self.args['args_w'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2, rtol=1e-2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fw_hess(self):
        """The _fw_hess function should return the hessian of the regularized sum squared error with respect to W."""  # noqa
        expected_0 = np.array(
            [6.624e+05, 0.01, 0.02, 8.606e+05, 0.04, 0.05, 6.624e+05, 0.07,
             0.08, 8.606e+05, 0.1, 0.11, 6.624e+05, 8.606e+05, 0.14, 1.987e+06,
             0.16, 0.17, 2.425e+06, 0.18, 0.2, 1.987e+06, 0.22, 0.23,
             2.425e+06, 0.25, 0.26, 1.987e+06, 2.425e+06, 0.29])
        hess_p = np.arange(30)
        actual_0 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0])
        np.testing.assert_allclose(
            expected_0, actual_0, rtol=1e-2, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0))
        actual_s = _fw_hess(self.input_w, hess_p, *self.args['args_ws'])
        np.testing.assert_allclose(
            expected_0, actual_s, rtol=1e-2, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_s))
        expected_1 = np.array(
            [662400., 0., 0., 860640., 0., 0., 662400., 0., 0., 860640., 0.,
             0., 662400., 860640., 0., 1987200., 0., 0., 2425440., 0., 0.,
             1987200., 0., 0., 2425440., 0., 0., 1987200., 2425440., 0.])
        actual_1 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1))
        expected_2 = np.array(
            [6.624e+05, 0.005, 0.01, 8.606e+05, 0.02, 0.025, 6.624e+05, 0.035,
             0.04, 8.606e+05, 0.05, 0.055, 6.624e+05, 8.606e+05, 0.07,
             1.987e+06, 0.08, 0.085, 2.425e+06, 0.095, 0.1, 1.987e+06, 0.11,
             0.115, 2.425e+06, 0.125, 0.13, 1.987e+06, 2.425e+06, 0.145])
        actual_2 = _fw_hess(self.input_w, hess_p, *self.args['args_w'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2, rtol=1e-2, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2))

    def test_fit_imc(self):
        """The _fit_imc function should solve for W and H or just W depending on the value of `update_H`, return W, H and a result message."""  # noqa
        with self.assertRaises(ValueError) as context:
            _fit_imc(
                self.data['r'], self.data['x'], self.data['y'],
                n_components=0)
        expected_msg = 'Number of components must be a positive integer; ' +\
            'got (n_components=0).'
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        expected_msg = (
            'Desired error not necessarily achieved due to precision loss.'
        )
        expected_w = np.array(
            [[2.6249, 1.91e-5, 2.87e-5, 8.8955, 2.40e-5, 2.88e-5, 2.6249,
              3.85e-5, 4.33e-5, 8.8955, 5.29e-5, 5.77e-5, 2.625, 8.8955,
              7.22e-5],
             [7.8749, 8.18e-5, 8.66e-5, 25.0692, 9.63e-5, 1.01e-4, 7.8749,
              1.1e-4, 1.15e-4, 25.0692, 1.25e-4, 1.3e-4, 7.875, 25.0693,
              1.44e-4]])
        _, _, succ, msg = _fit_imc(
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
        actual_w, _, succ, msg = _fit_imc(
            self.data['r'], self.data['x'], self.data['y'], H=self.data['H'],
            W=self.data['W'], n_components=2, update_H=False)
        np.testing.assert_allclose(
            actual_w, expected_w, rtol=1e-2, atol=1e-2,
            err_msg='Expected {}, but found {}.'.format(expected_w, actual_w))

    def test_check_init(self):
        """The _check_init function should check to ensure that an array has a specified shape and is not all zeros, raising an error if not."""  # noqa
        with self.assertRaises(ValueError) as context:
            _check_init(self.data['H'], (3, 4), 'Check Init')
        expected_msgs = [
            'Array with wrong shape passed to Check Init. Expected (3, 4), ' +
            'but got (2, 20).',
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
        r_h, x_h, y_h = _format_data(
            self.data['R'], self.data['X'], self.data['Y'])
        self.assertEqual(
            r_h.shape, (4, ),
            msg='Expected {}, but found {}.'.format(r_h.shape, (4, )))
        self.assertEqual(
            x_h.shape, (4, 15),
            msg='Expected {}, but found {}.'.format(x_h.shape, (4, 3)))
        self.assertEqual(
            y_h.shape, (4, 20),
            msg='Expected {}, but found {}.'.format(y_h.shape, (4, 4)))

    def test_fit_transform(self):
        """The fit_transform method should call the _fit_imc method to fit the IMC, finally returning the W and H matrices and returning a warning if convergence is not achieved."""  # noqa
        expected_w = np.array(
            [[-0.5249, 1.076e-16, 0, -0.8931, 0, 0, -0.5249, 0, 0, -0.8931, 0,
              0, -0.5249, -0.8931, 0],
             [0.7753, 0, 0, 0.8659, 0, 0, 0.7753, 0, 0, 0.8659, 0, 0, 0.7753,
              0.8659, 0]])
        expected_h = np.array(
            [[-55.4986, 0, 0, 0, -956.8486, 0, -55.4986, 0, -956.8486, 0, 0, 0,
              -1012.3473, 0, 0, 0, -956.8486, 0, -55.4986, 0],
             [145.8458, 0, 0, 0, -27.2898, 0, 145.8458, 0, -27.2898, 0, 0, 0,
              118.556, 0, 0, 0, -27.2898, 0, 145.8458, 0]])
        actual_w, actual_h = self.imcs['imc0']\
            .fit_transform(self.data['R'], self.data['X'], self.data['Y'])
        np.testing.assert_allclose(
            expected_w, actual_w[:2], atol=1e-1, rtol=1e-2,
            err_msg='Expected {}, but found {}.'.format(
                expected_w, actual_w[:2]))
        np.testing.assert_allclose(
            expected_h, actual_h[:2], atol=1e-2, rtol=1e-1,
            err_msg='Expected {}, but found {}.'.format(
                expected_h, actual_h[:2]))
        expected_w = np.array(
            [[-0.5341, 1.05e-16, 0, -0.8937, 0, 0, -0.5341, 0, 0, -0.8937, 0,
              0, -0.5341, -0.8937, 0],
             [0.7735, 0, 0, 0.766, 0, 0, 0.7735, 0, 0, 0.766, 0, 0, 0.7735,
              0.766, 0]])
        expected_h = np.array(
            [[-54.4324, 0, 0, 0, -956.9354, 0, -54.4324, 0, -956.9354, 0, 0, 0,
              -1011.3678, 0, 0, 0, -956.9354, 0, -54.4324, 0],
             [142.6703, 0, 0, 0, -25.6562, 0, 142.6703, 0, -25.6562, 0, 0, 0,
              117.0141, 0, 0, 0, -25.6562, 0, 142.6703, 0]])
        actual_w, actual_h = self.imcs['imck']\
            .fit_transform(self.data['R'], self.data['X'], self.data['Y'])
        np.testing.assert_allclose(
            expected_w, actual_w, rtol=1e-2, atol=1e-2,
            err_msg='Expected {}, but found {}.'.format(expected_w, actual_w))
        np.testing.assert_allclose(
            expected_h, actual_h, rtol=1e-2, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected_h, actual_h))
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
        actual_h = hasattr(result, 'components_h')
        actual_w = hasattr(result, 'components_w')
        self.assertEqual(
            expected, actual_w,
            msg='Expected {}, but found {}.'.format(expected, actual_w))
        self.assertEqual(
            expected, actual_h,
            msg='Expected {}, but found {}.'.format(expected, actual_h))

    def test_transform(self):
        """The transform method should return the W matrix constructed from the fitted model."""  # noqa
        with self.assertRaises(NotFittedError) as context:
            self.imcs['imc0'].transform(
                self.data['R'], self.data['X'], self.data['Y'])
        expected_msg = "This IMC instance is not fitted yet. Call 'fit' " +\
            "with appropriate arguments before using this method."
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        expected = np.array([
            -0.7286, 6.062e-25, 0, -0.9013, 0, 0, -0.7285, 0, 0, -0.9013, 0,
            0, -0.7285, -0.9013, 0])
        self.imcs['imc0'].fit(self.data['R'], self.data['X'], self.data['Y'])
        actual = self.imcs['imc0'].transform(
            self.data['R'], self.data['X'], self.data['Y'])[0]
        np.testing.assert_allclose(
            expected, actual, rtol=1e-1, atol=1e-1,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    # def test_predict_one(self):
    #     """The predict_one method should return the predicted rating for a given user, course pair."""  # noqa
    #     pass
    #
    # def test_predict_all(self):
    #     """The predict_all method should return the predicted ratings for all courses for a given user."""  # noqa
    #     pass
    #
    # def score(self):
    #     """Score method should return the root mean squared error for the reconstructed matrix."""  # noqa
    #     pass
