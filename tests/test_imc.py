"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import sys
import unittest

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.exceptions import NotFittedError

from src.imc import (DataHolder, IMC, _check_init, _check_x, _cost, _cost_hess,
                     _cost_prime, _fit_imc, _format_data)


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
        z_shape = (15, 20)
        args_w = {
            key: (h_component, x_h, y_h, b_h, 0.01, l1, w_component.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])}
        args_cost = (x_h, y_h, b_h, 0.01, z_shape)
        args_sparse = (
            csc_matrix(x_h), csc_matrix(y_h), b_h, 0.01, z_shape)
        args_ws = (
            h_component, csc_matrix(x_h), csc_matrix(y_h), b_h, 0.01, 0,
            w_component.shape)
        self.args = {
            'args_w': args_w, 'args_cost': args_cost, 'args_s': args_sparse,
            'args_ws': args_ws}
        self.input_cost = np.zeros(300)
        self.input_w = w_component.flatten()

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_cost(self):
        """The _cost function should return the regularized sum squared error."""  # noqa
        expected = 130545000.0
        actual = _cost(self.input_cost, *self.args['args_cost'])
        self.assertEqual(
            expected, actual,
            msg='Expected {}, but found {}.'.format(expected, actual))

    def test_cost_prime(self):
        """The _cost_prime function should return the gradient of the regularized sum squared error with respect to W and H."""  # noqa
        expected = np.array(
            [-6300, 0, 0, 0, 0, 0, -6300, 0, 0, 0])
        actual = _cost_prime(self.input_cost, *self.args['args_cost'])[:10]
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_cost_hess(self):
        """The _cost_hess function should return the hessian of the regularized sum squared error with respect to W and H."""  # noqa
        expected = np.array(
            [3096.01, 0.01, 0.01, 0.01, 0.01, 0.01, 3096.01, 0.01, 0.01, 0.01])
        hess_p = np.arange(300)
        actual = _cost_hess(
            self.input_cost, hess_p, *self.args['args_cost'])[:10]
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

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
            'Optimization terminated successfully.'
        )
        expected = np.array(
            [261.41078838, -8.549312e-13, -7.525747e-29, -3.157136e-62,
             3.585063e-13, -3.2061835e-92, 261.41078838, 3.2943254e-125,
             3.585063e-13, -3.58996873e-158])
        _, succ, msg = _fit_imc(
            self.data['r'], self.data['x'], self.data['y'], verbose=1)
        actual_msg = sys.stdout.getvalue().split('\n')[0]
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        self.assertEqual(
            True, succ,
            msg='Expected {}, but found {}.'.format(True, succ))
        self.assertEqual(
            expected_msg, msg,
            msg='Expected {}, but found {}.'.format(expected_msg, msg))
        actual, succ, msg = _fit_imc(
            self.data['r'], self.data['x'], self.data['y'])
        actual = actual[0, :10]
        np.testing.assert_allclose(
            actual, expected,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

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

    def test_check_x(self):
        """The _check_x function should ensure that the passed object is either a tuple or a DataHolder and output correctly formatted data for use by the predict for fit methods."""  # noqa
        with self.assertRaises(TypeError) as context:
            _check_x(self.data['X'])
        expected_msg = "Type of argument X should be tuple or DataHolder, " +\
            "was numpy.ndarray."
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        with self.assertRaises(ValueError) as context:
            _check_x((self.data['X'], ))
        expected_msg = 'Argument X should be a tuple of length 2 containing' +\
            ' an array for user attributes and an array for item attributes.'
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        holder = DataHolder(self.data['X'], self.data['Y'])
        actual_x, actual_y = _check_x(holder)
        expected_x = self.data['X']
        expected_y = self.data['Y']
        np.testing.assert_array_equal(
            expected_x, actual_x,
            err_msg='Expected {}, but found {}.'.format(expected_x, actual_x))
        np.testing.assert_array_equal(
            expected_y, actual_y,
            err_msg='Expected {}, but found {}.'.format(expected_y, actual_y))

    def test_fit_transform(self):
        """The fit_transform method should call the _fit_imc method to fit the IMC, finally returning the W and H matrices and returning a warning if convergence is not achieved."""  # noqa
        expected = np.array(
            [262.390671, -9.4394948e-13, -8.30935225e-29, -3.48586706e-62,
             3.95835182e-13, -3.54002186e-92, 262.390671, 3.63734141e-125,
             3.95835182e-13, -3.96376806e-158])
        actual = self.imcs['imc0']\
            .fit_transform((self.data['X'], self.data['Y']), self.data['R'])[0]
        np.testing.assert_allclose(
            expected, actual[:10],
            err_msg='Expected {}, but found {}.'.format(expected, actual[:10]))
        expected = np.array(
            [261.41078838, -8.549312e-13, -7.525747e-29, -3.157136e-62,
             3.585063e-13, -3.2061835e-92, 261.41078838, 3.2943254e-125,
             3.585063e-13, -3.58996873e-158])
        actual = self.imcs['imck']\
            .fit_transform((self.data['X'], self.data['Y']), self.data['R'])[0]
        np.testing.assert_allclose(
            expected, actual[:10],
            err_msg='Expected {}, but found {}.'.format(expected, actual[:10]))

    def test_fit(self):
        """The fit method should call the fit_transform method and return self."""  # noqa
        expected = True
        result = self.imcs['imc0']\
            .fit((self.data['X'], self.data['Y']), self.data['R'])
        actual_z = hasattr(result, 'Z')
        actual_err = hasattr(result, 'reconstruction_err_')
        self.assertEqual(
            expected, actual_z,
            msg='Expected {}, but found {}.'.format(expected, actual_z))
        self.assertEqual(
            expected, actual_err,
            msg='Expected {}, but found {}.'.format(expected, actual_err))
        expected_err = -2277.336256013376
        err = result.reconstruction_err_
        np.testing.assert_allclose(
            expected_err, err,
            err_msg='Expected {}, but found {}.'.format(expected_err, err))

    # def test_transform(self):
    #     """The transform method should return the W matrix constructed from the fitted model."""  # noqa
    #     with self.assertRaises(NotFittedError) as context:
    #         self.imcs['imc0'].transform(
    #             (self.data['X'], self.data['Y']), self.data['R'])
    #     expected_msg = "This IMC instance is not fitted yet. Call 'fit' " +\
    #         "with appropriate arguments before using this method."
    #     actual_msg = str(context.exception)
    #     self.assertEqual(
    #         expected_msg, actual_msg,
    #         msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
    #     expected = np.array([
    #         -0.1528, 5.285e-25, 0, -0.1222, 0, 0, -0.1528, 0, 0, -0.1222, 0,
    #         0, -0.1528, -0.1222, 0])
    #     self.imcs['imc0'].fit((self.data['X'], self.data['Y']), self.data['R'])
    #     actual = self.imcs['imc0'].transform(
    #         (self.data['X'], self.data['Y']), self.data['R'])[0]
    #     np.testing.assert_allclose(
    #         expected, actual, rtol=1e-1, atol=1e-1,
    #         err_msg='Expected {}, but found {}.'.format(expected, actual))
    #
    def test_predict_one(self):
        """The predict_one method should call the predict method with the given user/item pair."""  # noqa
        with self.assertRaises(NotFittedError) as context:
            self.imcs['imc0'].predict_one(
                (self.data['X'][0], self.data['Y'][0]))
        expected_msg = "This IMC instance is not fitted yet. Call 'fit' " +\
            "with appropriate arguments before using this method."
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        expected = np.array([[3148.68804665]])
        self.imcs['imc0'].fit((self.data['X'], self.data['Y']), self.data['R'])
        actual = self.imcs['imc0'].predict_one(
            (self.data['X'][0], self.data['Y'][0]))
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_predict(self):
        """The _predict method should return the predicted ratings for all given user/item pairs."""  # noqa
        expected = np.array([[3148.68804665]])
        self.imcs['imc0'].fit((self.data['X'], self.data['Y']), self.data['R'])
        actual = self.imcs['imc0']._predict(
            self.data['X'][[0]], self.data['Y'][[0]])
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_predict_all(self):
        """The predict_all method should call the _predict method with the given user/item pairs."""  # noqa
        expected = np.array([
            [3148.68804665, 787.17201166],
            [2623.90670554, 10495.62682216]])
        self.imcs['imc0'].fit((self.data['X'], self.data['Y']), self.data['R'])
        actual = self.imcs['imc0'].predict_all(
            (self.data['X'], self.data['Y']))
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_score(self):
        """Score method should return the root mean squared error for the reconstructed matrix."""  # noqa
        self.imcs['imc0'].fit((self.data['X'], self.data['Y']), self.data['R'])
        expected = 5.748504798089005
        actual = self.imcs['imc0'].score(
            (self.data['X'], self.data['Y']), self.data['R'])
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))
