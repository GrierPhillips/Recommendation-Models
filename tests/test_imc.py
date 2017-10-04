"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from src.imc import IMC, _cost, _cost_hess, _cost_prime, _fit_imc


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
        """Set up the variables needed for the tests."""
        self.imcs = {
            'imc0': IMC(alpha=0.01),
            'imck': IMC(n_components=2)}
        user_atts = np.array([
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
        item_atts = np.array([
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
        ratings_mat = csr_matrix(np.array([[300, 6000], [9000, 12000]]))
        rows, cols = ratings_mat.nonzero()
        x_h = user_atts[rows]
        y_h = item_atts[cols]
        b_h = ratings_mat.data
        self.data = {'r': b_h, 'x': x_h, 'y': y_h, 'R': ratings_mat,
                     'X': user_atts, 'Y': item_atts}
        self.args = (x_h, y_h, b_h, 0.01, (15, 20))
        self.input_cost = np.zeros(300)

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_cost(self):
        """The _cost function should return the regularized sum squared error."""  # noqa
        expected = 130545000.0
        actual = _cost(self.input_cost, *self.args)
        self.assertEqual(
            expected, actual,
            msg='Expected {}, but found {}.'.format(expected, actual))

    def test_cost_prime(self):
        """The _cost_prime function should return the gradient of the regularized sum squared error with respect to W and H."""  # noqa
        expected = np.array(
            [-300, 0, 0, 0, -6000, 0, -300, 0, -6000, 0])
        actual = _cost_prime(self.input_cost, *self.args)[:10]
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_cost_hess(self):
        """The _cost_hess function should return the hessian of the regularized sum squared error with respect to W and H."""  # noqa
        expected = np.array(
            [1.54801e3, 0.01, 0.01, 0.01, 1.56001e3, 0.01, 1.54801e3, 0.01,
             1.56001e3, 0.01])
        hess_p = np.arange(300)
        actual = _cost_hess(
            self.input_cost, hess_p, *self.args)[:10]
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_fit_imc(self):
        """The _fit_imc function should solve for W and H or just W depending on the value of `update_H`, return W, H and a result message."""  # noqa
        with self.assertRaises(ValueError) as context:
            _fit_imc(
                self.data['x'], self.data['y'], self.data['r'],
                n_components=0)
        expected_msg = 'Number of components must be a positive integer; ' +\
            'got (n_components=0).'
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        expected_msg = (
            'Warning: Desired error not necessarily achieved due to precision'
            ' loss.'
        )
        expected = np.array(
            [-1.04576367e2, -1.35697449e-3, -1.35697449e-3, -1.35697449e-3,
             5.21791002e2, -1.35697449e-3, -1.04576367e2, -1.35697449e-3,
             5.21791002e2, -1.35697449e-3])
        _, succ, msg = _fit_imc(
            self.data['x'], self.data['y'], self.data['r'], verbose=1)
        actual_msg = sys.stdout.getvalue().split('\n')[0]
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        self.assertEqual(
            False, succ,
            msg='Expected {}, but found {}.'.format(False, succ))
        self.assertEqual(
            expected_msg, msg,
            msg='Expected {}, but found {}.'.format(expected_msg, msg))
        actual, succ, msg = _fit_imc(
            self.data['x'], self.data['y'], self.data['r'])
        actual = actual[0, :10]
        np.testing.assert_allclose(
            actual, expected,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_fit_transform(self):
        """The fit_transform method should call the _fit_imc method to fit the IMC, finally returning the W and H matrices and returning a warning if convergence is not achieved."""  # noqa
        expected = np.array(
            [-1.06455138e2, -1.02742138e-13, 4.89509721e-30, -1.11717223e-59,
             5.26175380e2, 1.25470694e-91, -1.06455138e2, -7.13882988e-125,
             5.26175380e2, 4.16328484e-157])
        actual = self.imcs['imc0']\
            .fit_transform((self.data['X'], self.data['Y']), self.data['R'])[0]
        np.testing.assert_allclose(
            expected, actual[:10],
            err_msg='Expected {}, but found {}.'.format(expected, actual[:10]))
        expected = np.array(
            [-1.0457636e2, -1.35697449e-3, -1.35697449e-3, -1.35697449e-3,
             5.217910e2, -1.35697449e-3, -1.0457636e2, -1.35697449e-3,
             5.217910e2, -1.35697449e-3])
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
        expected_err = 5.748504798089005
        err = result.reconstruction_err_
        np.testing.assert_allclose(
            expected_err, err,
            err_msg='Expected {}, but found {}.'.format(expected_err, err))

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
        expected = 301.06449075427315
        self.imcs['imc0'].fit((self.data['X'], self.data['Y']), self.data['R'])
        actual = self.imcs['imc0'].predict_one(
            (self.data['X'][0], self.data['Y'][0]))
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_predict(self):
        """The _predict method should return the predicted ratings for all given user/item pairs."""  # noqa
        expected = 301.06449075427315
        self.imcs['imc0'].fit((self.data['X'], self.data['Y']), self.data['R'])
        actual = self.imcs['imc0']._predict(
            (self.data['X'][[0]], self.data['Y'][[0]]))
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_predict_all(self):
        """The predict_all method should call the _predict method with the given user/item pairs."""  # noqa
        expected = np.array([
            [301.06449075, 5994.73915285],
            [8994.67063528, 11991.34151006]])
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
