"""Tests for the ALS class."""

# pylint: disable=C0301

import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src import ALS


class ALSTest(unittest.TestCase):
    """Suite of tests for the ALS class."""

    def setUp(self):
        """Set up the variables needed for the tests."""
        ratings_mat = csr_matrix(np.array([[3, 0, 4], [4, 5, 0]]))
        users, items = ratings_mat.nonzero()
        self.data = {'users': users, 'items': items, 'R': ratings_mat,
                     'r': ratings_mat.data}
        self.als = ALS(2, tol=0.000001, random_state=42)
        self.fitted = ALS(2, tol=0.000001, random_state=42)
        self.fitted.fit((self.data['users'], self.data['items']),
                        self.data['R'])
        self.bad = ALS(2, n_jobs=sys.maxsize)

    def test_fit(self):
        """The fit method should call the fit_transform method and return self."""  # noqa
        expected = True
        self.als.fit((self.data['users'], self.data['items']), self.data['R'])
        actual = hasattr(self.als, 'reconstruction_err_')
        self.assertEqual(
            expected, actual,
            msg='Expected {}, but found {}.'.format(expected, actual))
        expected_err = -0.10663924910933506
        actual_err = self.als.reconstruction_err_
        np.testing.assert_allclose(
            expected_err, actual_err,
            err_msg='Expected {}, but found {}.'.format(
                expected_err, actual_err))

    def test_fit_transform(self):
        """The fit_transform method should make sure the data is properly formatted and execute the fit_als.py script, finally returning the user and item features."""  # noqa
        expected_u = np.array(
            [[1.52198247, 1.97980245],
             [-0.79543119, -1.03470076]])
        users, items = self.als.fit_transform(
            (self.data['users'], self.data['items']), self.data['R'])
        np.testing.assert_allclose(
            expected_u, users,
            err_msg='Expected {}, but found {}.'.format(expected_u, users))
        expected_i = np.array(
            [[1.5339228, 1.94471076, 1.99660463],
             [-0.80167146, -1.01636085, -1.04348219]])
        np.testing.assert_allclose(
            expected_i, items,
            err_msg='Expected {}, but found {}.'.format(expected_i, items))
        with self.assertRaises(ValueError) as context:
            self.als.fit_transform(
                (self.data['users'], self.data['items']), self.data['r'])
        expected_msg = 'When y is a scalar or 1-D array shape must be' +\
            'provided.'
        actual_msg = str(context.exception)
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))
        users, items = self.als.fit_transform(
            (self.data['users'], self.data['items']), self.data['r'],
            shape=self.data['R'].shape)
        np.testing.assert_allclose(
            expected_u, users,
            err_msg='Expected {}, but found {}.'.format(expected_u, users))
        np.testing.assert_allclose(
            expected_i, items,
            err_msg='Expected {}, but found {}.'.format(expected_i, items))
        with self.assertRaises(ValueError) as context:
            self.bad.fit_transform(
                (self.data['users'], self.data['items']), self.data['R'])
        expected_msg = 'Fitting ALS failed with error:'
        actual_msg = str(context.exception).split('\n')[0]
        self.assertEqual(
            expected_msg, actual_msg,
            msg='Expected {}, but found {}.'.format(expected_msg, actual_msg))

    def test_predict(self):
        """The _predict method should take a tuple or DataHolder ensure the data is formatted properly and calculate the predictions for the given user/item pairs."""  # noqa
        expected = np.array([2.9722781, 3.86881553, 3.86635418, 4.90177248])
        actual = self.fitted._predict((self.data['users'], self.data['items']))
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_predict_one(self):
        """The predict_one method should take a single user and item and return the predicted value for that entry."""  # noqa
        expected = 3.768260826602256
        actual = self.fitted.predict_one(0, 1)
        self.assertEqual(
            expected, actual,
            msg='Expected {}, but found {}.'.format(expected, actual))

    def test_predict_all(self):
        """The predict_all method should take a user index and return all of the predictions for that user."""  # noqa
        expected = np.array([2.972278, 3.76826083, 3.86881553])
        actual = self.fitted.predict_all(0)
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_score(self):
        """The score method should take a tuple or DataHolder containing user and item indices and true values for those indices and return the root mean squared error between the predicted values and the true values."""  # noqa
        expected = -0.10663924910933506
        actual = self.fitted.score((self.data['users'], self.data['items']),
                                   self.data['r'])
        self.assertEqual(
            expected, actual,
            msg='Expected {}, but found {}.'.format(expected, actual))

    def test_update_user(self):
        """The update_user method should take a user, item, and value as input, update the data matrix and update the features for the given user based on the new input."""  # noqa
        expected = np.array(
            [[1.69210752, 1.97980245],
             [-0.8843431, -1.03470076]])
        self.fitted.update_user(0, 1, 5)
        actual = self.fitted.user_feats
        np.testing.assert_allclose(
            expected, actual,
            err_msg='Expected {}, but found {}.'.format(expected, actual))

    def test_add_user(self):
        """The add_user method should append a row of zeros to the end of the data matrix and a column of zeros to the user_feats."""  # noqa
        expected = 3
        self.fitted.add_user()
        actual_r = self.fitted.data.shape[0]
        self.assertEqual(
            expected, actual_r,
            msg='Expected {}, but found {}.'.format(expected, actual_r))
        actual_c = self.fitted.user_feats.shape[1]
        self.assertEqual(
            expected, actual_c,
            msg='Expected {}, but found {}.'.format(expected, actual_c))
