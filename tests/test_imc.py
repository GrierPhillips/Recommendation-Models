"""Tests for the IMC class."""

# pylint: disable=W0212,C0301

import unittest

import numpy as np

from src.imc import IMC


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
        self.imc0 = IMC(max_iter=20, alpha=0.01, l1_ratio=0)
        self.imc1 = IMC(max_iter=20, alpha=0.01, l1_ratio=1)
        self.imc0_5 = IMC(max_iter=20, alpha=0.01, l1_ratio=0.5)
        item_att_feats = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.user_atts = np.array([[1, 2, 3], [4, 5, 6]])
        self.item_atts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        user_att_feats = np.array([[1, 2, 3], [4, 5, 6]])
        self.ratings = np.array([[300, 6000], [9000, 12000]])
        self.args_h = {
            key: (
                user_att_feats, self.user_atts, self.item_atts, self.ratings,
                0.01, l1, item_att_feats.shape
            ) for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        self.args_w = {
            key: (
                item_att_feats, self.user_atts, self.item_atts, self.ratings,
                0.01, l1, user_att_feats.shape
            ) for (key, l1) in zip([0, 1, 0.5], [0, 1, 0.5])
        }
        self.input_h = item_att_feats.flatten()
        self.input_w = user_att_feats.flatten()

    def tearDown(self):
        """Teardown the IMC class after each test."""
        pass

    def test_fit(self):
        """The fit method should call the fit_transform method and return self."""  # noqa
        expected = True
        result = self.imc0.fit(self.ratings, self.user_atts, self.item_atts)
        actual = hasattr(result, 'components_')
        self.assertEqual(
            expected,
            actual,
            msg='Expecte {}, but found {}.'.format(expected, actual)
        )


    def test_fit_transform(self):
        """The fit_transform method should iterate through the algorithm until the fit condition is met, finally returning the W matrix."""  # noqa
        expected_0 = np.array(
            [[-0.83361051, -0.15117396, 0.53126259],
             [-0.37205758, -0.55720711, -0.74235665],
             [0.40824829, -0.81649658, 0.40824829]]
        )
        actual_0 = self\
            .imc0.fit_transform(self.ratings, self.user_atts, self.item_atts)
        np.testing.assert_allclose(
            expected_0,
            actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )

    def test_fh(self):
        """The _fh function should return the regularized sum squared error."""
        expected_0 = 13063724.476444978
        actual_0 = self.imc0._fh(self.input_h, *self.args_h[0])
        self.assertEqual(
            expected_0,
            actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = 13063724.84
        actual_1 = self.imc1._fh(self.input_h, *self.args_h[1])
        self.assertEqual(
            expected_1,
            actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = 13063724.658222489
        actual_2 = self.imc0_5._fh(self.input_h, *self.args_h[0.5])
        self.assertEqual(
            expected_2,
            actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fh_prime(self):
        """The _fh_prime function should return the gradient of the regularized sum squared error with respect to V."""  # noqa
        expected_0 = np.array(
            [568680.01, 641008.02, 713336.03, 785664.04, 1359780.05,
             1528912.06, 1698044.07, 1867176.08]
        )
        actual_0 = self.imc0._fh_prime(self.input_h, *self.args_h[0])
        np.testing.assert_array_equal(
            expected_0,
            actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [568680.005, 641008.005, 713336.005, 785664.005, 1359780.005,
             1528912.005, 1698044.005, 1867176.005]
        )
        actual_1 = self.imc1._fh_prime(self.input_h, *self.args_h[1])
        np.testing.assert_array_equal(
            expected_1,
            actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [568680.0075, 641008.0125, 713336.0175, 785664.0225, 1359780.0275,
             1528912.0325, 1698044.0375, 1867176.0425]
        )
        actual_2 = self.imc0_5._fh_prime(self.input_h, *self.args_h[0.5])
        np.testing.assert_allclose(
            expected_2,
            actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fh_hess(self):
        """The _fh_hess function should return the hessian of the regularized sum squared error with respect to V."""  # noqa
        expected_0 = np.array(
            [2622400., 3306176.01, 3989952.02, 4673728.03, 6261280.04,
             7893872.05, 9526464.06, 11159056.07]
        )
        hess_p = np.arange(8)
        actual_0 = self.imc0._fh_hess(self.input_h, hess_p, *self.args_h[0])
        np.testing.assert_array_equal(
            expected_0,
            actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [2622400., 3306176., 3989952., 4673728., 6261280.,
             7893872., 9526464., 11159056.]
        )
        actual_1 = self.imc1._fh_hess(self.input_h, hess_p, *self.args_h[1])
        np.testing.assert_array_equal(
            expected_1,
            actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [2622400., 3306176.005, 3989952.01, 4673728.015, 6261280.02,
             7893872.025, 9526464.03, 11159056.035]
        )
        actual_2 = self.\
            imc0_5._fh_hess(self.input_h, hess_p, *self.args_h[0.5])
        np.testing.assert_array_equal(
            expected_2,
            actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw(self):
        """The _fw function should return the regularized sum squared error."""
        expected_0 = 13063724.476444978
        actual_0 = self.imc0._fw(self.input_w, *self.args_w[0])
        self.assertEqual(
            expected_0,
            actual_0,
            msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = 13063724.84
        actual_1 = self.imc1._fw(self.input_w, *self.args_w[1])
        self.assertEqual(
            expected_1,
            actual_1,
            msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = 13063724.658222489
        actual_2 = self.imc0_5._fw(self.input_w, *self.args_w[0.5])
        self.assertEqual(
            expected_2,
            actual_2,
            msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw_prime(self):
        """The _fw_prime function should return the gradient of the regularized sum squared error with respect to U."""  # noqa
        expected_0 = np.array(
            [809800.01, 1094120.02, 1378440.03, 2050600.04, 2758664.05,
             3466728.06]
        )
        actual_0 = self.imc0._fw_prime(self.input_w, *self.args_w[0])
        np.testing.assert_array_equal(
            expected_0,
            actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [809800.01, 1094120.01, 1378440.01, 2050600.01, 2758664.01,
             3466728.01]
        )
        actual_1 = self.imc1._fw_prime(self.input_w, *self.args_w[1])
        np.testing.assert_array_equal(
            expected_1,
            actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [809800.01, 1094120.015, 1378440.02, 2050600.025, 2758664.03,
             3466728.035]
        )
        actual_2 = self.imc0_5._fw_prime(self.input_w, *self.args_w[0.5])
        np.testing.assert_allclose(
            expected_2,
            actual_2,
            err_msg='Expected {}, but found {}.'.format(expected_2, actual_2)
        )

    def test_fw_hess(self):
        """The _fw_hess function should return the hessian of the regularized sum squared error with respect to U."""  # noqa
        expected_0 = np.array(
            [4353520., 5755160.01, 7156800.02, 10723504.03, 14175992.04,
             17628480.05]
        )
        hess_p = np.arange(6)
        actual_0 = self.imc0._fw_hess(self.input_w, hess_p, *self.args_w[0])
        np.testing.assert_array_equal(
            expected_0,
            actual_0,
            err_msg='Expected {}, but found {}.'.format(expected_0, actual_0)
        )
        expected_1 = np.array(
            [4353520., 5755160., 7156800., 10723504., 14175992., 17628480.]
        )
        actual_1 = self.imc1._fw_hess(self.input_w, hess_p, *self.args_w[1])
        np.testing.assert_array_equal(
            expected_1,
            actual_1,
            err_msg='Expected {}, but found {}.'.format(expected_1, actual_1)
        )
        expected_2 = np.array(
            [4353520., 5755160.005, 7156800.01, 10723504.015, 14175992.02,
             17628480.025]
        )
        actual_2 = self.\
            imc0_5._fw_hess(self.input_w, hess_p, *self.args_w[0.5])
        np.testing.assert_array_equal(
            expected_2,
            actual_2,
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
