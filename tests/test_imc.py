"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import unittest


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
        pass

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
        pass

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
