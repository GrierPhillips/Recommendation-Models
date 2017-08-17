"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import sys
import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.imc import (IMC, _fh, _fh_hess, _fh_prime,
                     _fit_inductive_matrix_completion, _fw, _fw_hess,
                     _fw_prime)


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
        ratings = np.array([[300, 6000], [9000, 12000]])
        self.data = {'R': ratings, 'X': X, 'Y': Y}
        args_h = {
            key: (W, X, Y, ratings, 0.01, l1, H.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        args_w = {
            key: (H, X, Y, ratings, 0.01, l1, W.shape)
            for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        self.args = {'args_h': args_h, 'args_w': args_w}
        self.input_h = H.flatten()
        self.input_w = W.flatten()

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_fit(self):
        """The fit method should call the fit_transform method and return self."""  # noqa
        expected = True
        result = self.imcs['imc0']\
            .fit(self.data['R'], self.data['X'], self.data['X'])
        actual = hasattr(result, 'components_')
        self.assertEqual(
            expected, actual,
            msg='Expecte {}, but found {}.'.format(expected, actual)
        )

    def test_fit_transform(self):
        """The fit_transform method should call the _fit_inductive_matrix_completion method to fit the IMC, finally returning the W matrix and returning a warning if convergence is not achieved."""  # noqa
        expected_0 = np.array(
            [[-0.833617, -0.15118367, 0.53124965],
             [-0.37204305, -0.55720448, -0.74236591],
             [0.40824829, -0.81649658, 0.40824829]]
        )
        actual_0 = self.imcs['imc0']\
            .fit_transform(self.data['R'], self.data['X'], self.data['X'])
        np.testing.assert_allclose(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [[-0.83600716, -0.15478636, 0.52643444],
             [-0.36664065, -0.55621445, -0.74578825]]
        )
        actual_1 = self.imcs['imck']\
            .fit_transform(self.data['R'], self.data['X'], self.data['X'])
        np.testing.assert_allclose(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1),
        )
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always')
            self.imcs['imcw']\
                .fit_transform(self.data['R'], self.data['X'], self.data['X'])
            self.assertEqual(warn[0].category, ConvergenceWarning)

    def test_fit_imc(self):
        """The _fit_inductive_matrix_completion method should iterate through the AltMin-LRROM algorithm until convergence or max iterations is met, while printing messages in verbose mode."""  # noqa
        expected_msg = 'Loss: 50639'
        _fit_inductive_matrix_completion(
            self.data['R'], self.data['X'], self.data['X'], W=None, H=None,
            n_components=None, max_iter=1, verbose=1)
        actual_msg = sys.stdout.getvalue().strip().split('.')[0]
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))

    def test_fh(self):
        """The _fh function should return the regularized sum squared error."""
        expected_0 = 13063724.476444978
        actual_0 = _fh(self.input_h, *self.args['args_h'][0])
        self.assertEqual(
            expected_0, actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = 13063724.84
        actual_1 = _fh(self.input_h, *self.args['args_h'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = 13063724.658222489
        actual_2 = _fh(self.input_h, *self.args['args_h'][0.5])
        self.assertEqual(
            expected_2, actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fh_prime(self):
        """The _fh_prime function should return the gradient of the regularized sum squared error with respect to V."""  # noqa
        expected_0 = np.array(
            [568680.01, 641008.02, 713336.03, 785664.04, 1359780.05,
             1528912.06, 1698044.07, 1867176.08]
        )
        actual_0 = _fh_prime(self.input_h, *self.args['args_h'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [568680.005, 641008.005, 713336.005, 785664.005, 1359780.005,
             1528912.005, 1698044.005, 1867176.005]
        )
        actual_1 = _fh_prime(self.input_h, *self.args['args_h'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [568680.0075, 641008.0125, 713336.0175, 785664.0225, 1359780.0275,
             1528912.0325, 1698044.0375, 1867176.0425]
        )
        actual_2 = _fh_prime(self.input_h, *self.args['args_h'][0.5])
        np.testing.assert_allclose(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fh_hess(self):
        """The _fh_hess function should return the hessian of the regularized sum squared error with respect to V."""  # noqa
        expected_0 = np.array(
            [2622400., 3306176.01, 3989952.02, 4673728.03, 6261280.04,
             7893872.05, 9526464.06, 11159056.07]
        )
        hess_p = np.arange(8)
        actual_0 = _fh_hess(self.input_h, hess_p, *self.args['args_h'][0])
        np.testing.assert_array_equal(
            expected_0, actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [2622400., 3306176., 3989952., 4673728., 6261280.,
             7893872., 9526464., 11159056.]
        )
        actual_1 = _fh_hess(self.input_h, hess_p, *self.args['args_h'][1])
        np.testing.assert_array_equal(
            expected_1, actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [2622400., 3306176.005, 3989952.01, 4673728.015, 6261280.02,
             7893872.025, 9526464.03, 11159056.035]
        )
        actual_2 = _fh_hess(self.input_h, hess_p, *self.args['args_h'][0.5])
        np.testing.assert_array_equal(
            expected_2, actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw(self):
        """The _fw function should return the regularized sum squared error."""
        expected_0 = 13063724.476444978
        actual_0 = _fw(self.input_w, *self.args['args_w'][0])
        self.assertEqual(
            expected_0, actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = 13063724.84
        actual_1 = _fw(self.input_w, *self.args['args_w'][1])
        self.assertEqual(
            expected_1, actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = 13063724.658222489
        actual_2 = _fw(self.input_w, *self.args['args_w'][0.5])
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
