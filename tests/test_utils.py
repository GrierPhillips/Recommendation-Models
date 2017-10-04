"""Tests for the functions in utils."""

# pylint: disable=C0301

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.utils import (_check_x, DataHolder, _format_data,
                       root_mean_squared_error)


class UtilsTest(unittest.TestCase):
    """Suite of tests for the utils functions."""

    def setUp(self):
        """Set up the variables needed for the tests."""
        user_atts = np.array([
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
        item_atts = np.array([
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
        ratings_mat = csr_matrix(np.array([[300, 6000], [9000, 12000]]))
        users, items = ratings_mat.nonzero()
        x_h = user_atts[users]
        y_h = item_atts[items]
        b_h = ratings_mat.data
        self.data = {'r': b_h, 'x': x_h, 'y': y_h, 'users': users,
                     'items': items, 'X': user_atts, 'Y': item_atts,
                     'R': ratings_mat}

    def test_rmse(self):
        """The root_mean_squared_error function should take two arrays and return the root mean squared error between them."""  # noqa
        expected = 1
        test_vals = np.array([301, 6001, 9001, 12001])
        actual = root_mean_squared_error(test_vals, self.data['r'])
        self.assertEqual(
            expected, actual,
            msg='Expected {}, but found {}.'.format(expected, actual))

    def test_check_x(self):
        """The _check_x function should take a tuple or DataHolder of arrays and return the individual arrays."""  # noqa
        expected = (self.data['X'], self.data['Y'])
        actual = _check_x(expected)
        self.assertEqual(
            expected, actual,
            msg='Expected {}, but found {}.'.format(expected, actual))
        dh = DataHolder(self.data['X'], self.data['Y'])
        actual_dh = _check_x(dh)
        self.assertEqual(
            expected, actual_dh,
            msg='Expected {}, but found {}.'.format(expected, actual_dh))
        expected_ind = (self.data['X'], self.data['Y'], self.data['users'],
                        self.data['items'])
        dh_ind = DataHolder(self.data['X'], self.data['Y'], self.data['users'],
                            self.data['items'])
        actual_ind = _check_x(dh_ind, indices=True)
        np.testing.assert_equal(
            expected_ind, actual_ind,
            err_msg='Expected {}, but found {}.'.format(
                expected_ind, actual_ind))
        with self.assertRaises(TypeError) as context:
            _check_x([self.data['X'], self.data['Y']])
        expected_msg = 'Type of argument X should be tuple or DataHolder, ' +\
            'was list.'
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        with self.assertRaises(ValueError) as context:
            _check_x((self.data['X'], self.data['Y'], self.data['users']))
        expected_msg = 'Argument X should be a tuple of length 2.'
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))

    def test_format_data(self):
        """The _format_data function should take a tuple or DataHolder and optionally an array y and return the correct arrays for each input."""  # noqa
        expected = (self.data['x'], self.data['y'], self.data['r'])
        actual = _format_data((self.data['X'], self.data['Y']), self.data['R'])
        np.testing.assert_equal(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(
                expected, actual))
        expected = (self.data['x'], self.data['y'], self.data['r'])
        actual = _format_data((self.data['x'], self.data['y']),
                              self.data['R'].toarray())
        np.testing.assert_equal(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(
                expected, actual))

    def test_DataHolder(self):
        """The DataHolder class is inteded as a helper for implementing GridSearchCV. It should take arrays and implement a __getitem__ method that allows them to all be split by a cv.split method."""  # noqa
        expected = (self.data['X'][0], self.data['Y'][0])
        dh = DataHolder(self.data['X'], self.data['Y'])
        actual = dh[0]
        np.testing.assert_equal(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(
                expected, actual))
        expected = (self.data['X'][0], self.data['Y'][0],
                    self.data['users'][0], self.data['items'][0])
        dh = DataHolder(self.data['X'], self.data['Y'], self.data['users'],
                        self.data['items'])
        actual = dh[0]
        np.testing.assert_equal(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(
                expected, actual))
