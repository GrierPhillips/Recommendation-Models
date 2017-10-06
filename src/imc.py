"""
Implementation of Inductive Matric Completion.

Inductive Matrix Completion (IMC) is an algorithm for recommender systems with
side-information of users and items. The IMC formulation incorporates features
associated with rows (users) and columns (items) in matrix completion, so that
it enables predictions for users or items that were not seen during training,
and for which only features are known but no dyadic information (such as
ratings or linkages).
"""


import numbers
import warnings

import numpy as np
from numpy.linalg import norm, svd
import scipy.optimize as so
from scipy.sparse import diags
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from .utils import _check_x, root_mean_squared_error


INTEGER_TYPES = (numbers.Integral, np.integer)


def _cost(arr, *args):
    """Return the regularized sum squared error for given variables.

    Parameters
    ----------
    arr : array-like, shape (n_samples, )
        Flattened array to minimize.

    Returns
    -------
    sse : The sum squared error for the given array.

    """
    users, items, ratings, lam, shape = args
    z_arr = arr.reshape(shape)
    x_z = users.dot(z_arr)
    preds = np.array([x_z[i].dot(items[i]) for i in range(x_z.shape[0])])
    sse = 0.5 * np.sum((ratings - preds) ** 2 + lam * norm(z_arr) ** 2)
    return sse


def _cost_prime(arr, *args):
    """Return the gradient of the regularized SSE for given array.

    Parameters
    ----------
    arr : array-like, shape (n_samples, )
        Flattened array to minimize.

    Returns
    -------
    grad_z : The gradient of the regularized sum squared error for the given array.

    """
    users, items, ratings, lam, shape = args
    z_arr = arr.reshape(shape)
    x_z = users.dot(z_arr)
    diag = np.array([x_z[i].dot(items[i]) for i in range(x_z.shape[0])])
    x_r = np.multiply(users.T, ratings)
    grad_z = users.T.dot(diags(diag).dot(items)) - x_r.dot(items) + lam * z_arr
    return grad_z.ravel()


def _cost_hess(_, s_vec, *args):
    """Return the hessian of the regularized SSE for given array.

    Parameters
    ----------
    arr : array-like, shape (n_samples, )
        Flattened array to minimize.

    s_vec : array-like, shape (n_samples, )
        Vector used to multiply the hessian.

    Returns
    -------
    hess_z : The hessian of the regularized sum squared error for the given
        array.

    """
    users, items, _, lam, shape = args
    s_vec = s_vec.reshape(shape)
    x_s = users.dot(s_vec)
    diag = np.array([x_s[i].dot(items[i]) for i in range(x_s.shape[0])])
    hess_z = users.T.dot(diags(diag).dot(items)) + lam
    return hess_z.ravel()


def _fit_imc(X, Y, R, Z=None, method='Newton-CG', n_components=None,
             alpha=0.1, verbose=0):
    """Compute Inductive Matrix Completion (IMC) with AltMin-LRROM.

    The objective function is minimized by updating Z in the formula R ~ XZY^T.

    Parameters
    ----------
    R : {array-like, sparse matrix} shape (n_samples, m_samples)
        Constant matrix where rows correspond to users and columns to items.

    X : array-like, shape (n_samples, p_attributes)
        Constant user attribute matrix.

    Y : array-like, shape (m_samples, q_attributes)
        Constant item attribute matrix.

    Z : array-like, shape (p_attributes, q_attributes)
        Initial guess for the solution.

    method : None | 'Newton-CG' | 'BFGS' (default='Newton-CG')
        Algorithm used to find W and H that minimize the cost function.

    n_components : int
        Number of components.

    alpha : double (default=0.1)
        Constant that multiplies the regularization terms.

    verbose : integer (default=0)
        The verbosity level.

    Returns
    -------
    Z : array-like, shape (p_attributes, q_attributes)
        Solution to the least squares problem.

    res.success : boolean
        Whether or not the minimization converged.

    res.message : string
        The message returned by the minimization process.

    References
    ----------
    Jain, Prateek and Dhillon, Inderjit S. "Provable Inductive Matrix
    Completion." arXiv1306.0626v1 [cs.LG], 2013.

    """
    _, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)." % n_components)

    if Z is None:
        sum_x_r_y = diags(R).T.dot(X).T.dot(Y) / R.size
        u, s, v = svd(sum_x_r_y, False)
        Z = u.dot(np.diag(s)).dot(v)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        x0 = Z.flatten()
        res = so.minimize(
            _cost, x0, method=method, jac=_cost_prime, hessp=_cost_hess,
            args=(X, Y, R, alpha, Z.shape), options={'disp': verbose})
        Z = res['x'].reshape(Z.shape)

    return Z, res.success, res.message


class IMC(BaseEstimator):
    """Implementation of Inductive Matrix Completion.

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set or is greater than the
        number of features in the item attributes matrix, all features are
        kept.

    method : None | 'Newton-CG' | 'BFGS' (default='Newton-CG')
        Algorithm used to find W and H that minimize the cost function.

    alpha : double (default=0.1)
        Constant that multiplies the regularization terms. Set to zero to have
        no regularization.

    l1_ratio : double (default=0)
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    verbose : int (default=0)
        The verbosity level.

    Attributes
    ----------
    Z : array, shape (p_attributes, q_attributes)
        IMC decomposition of the data. This is the Z matrix in the formula:
        ``R ~= XZY``.

    reconstruction_err_ : float
        The sum squared error between the values predicted by the model and the
        real values of the training data.

    """

    def __init__(self, n_components=None, method='Newton-CG', alpha=0.1,
                 verbose=0):
        """Initialize instance of IMC."""
        self.n_components = n_components
        self.method = method
        self.alpha = alpha
        self.verbose = verbose

    def fit_transform(self, X, y, Z=None):
        """Learn IMC model for given data and return decomposition.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing matrices of user attributes and item attributes.

        y : {array-like, sparse matrix}, shape (n_samples, m_samples)
            Data matrix to be decomposed.

        Z : array-like, shape (p_attributes, q_attributes)
            Initial guess for the IMC solution.

        Returns
        -------
        W : array, shape (min(self.n_components, q_attributes), p_attributes)

        H : array, shape (min(self.n_components, q_attributes), q_attributes)

        """
        x, y_ = _check_x(X)
        r = y
        if self.n_components and self.n_components < x.shape[1]:
            self.n_components_ = self.n_components
        else:
            self.n_components_ = x.shape[1]
        Z, success, msg = _fit_imc(
            x, y_, r, Z=Z, n_components=self.n_components_, method=self.method,
            alpha=self.alpha, verbose=self.verbose)
        if not success:
            warnings.warn(msg, ConvergenceWarning, stacklevel=1)
        self.Z = Z
        self.reconstruction_err_ = self.score(X, y)
        return Z

    def fit(self, X, y, Z=None):
        """Learn IMC model for the data R, with attribute data X and Y.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing matrices of user attributes and item attributes.

        y : {array-like, sparse matrix}, shape (n_samples, m_samples)
            Data matrix to be decomposed.

        Z : array-like, shape (p_attributes, q_attributes)
            Initial guess for the IMC solution.

        Returns
        -------
        self

        """
        _ = self.fit_transform(X, y, Z=Z)
        return self

    def _predict(self, X):
        """Make predictions for the given attribute arrays.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing matrices of user attributes and item attributes.

        Returns
        -------
        prediction : {float, array}, shape (n_samples, m_samples)
            Array of predicted values for the user/items pairs.

        """
        X, Y = _check_x(X)
        x_z = X.dot(self.Z)
        prediction = x_z.dot(Y.T)
        return prediction

    def predict_one(self, X):
        """Get the prediction for a single user/item pair.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing matrices of user attributes and item attributes.

        Returns
        -------
        prediction : float
            Predicted value for the user/item pair.

        """
        check_is_fitted(self, 'n_components_')
        prediction = self._predict(X)
        return prediction

    def predict_all(self, X):
        """Get the predictions for all combinations of user/item pairs.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing matrices of user attributes and item attributes.

        Returns
        -------
        predictions : array, shape (n_samples, m_samples)
            Predicted values for the user/item pairs.

        """
        check_is_fitted(self, 'n_components_')
        predictions = self._predict(X)
        return predictions

    def score(self, X, y):
        """Return the root mean squared error of the reconstructed matrix.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing matrices of user attributes and item attributes.

        y : {array-like, sparse matrix}, shape (n_samples, m_samples)
            Data matrix to be decomposed.

        Returns
        -------
        rmse : float
            The root mean squared error of the reconstructed matrix.

        """
        check_is_fitted(self, 'n_components_')
        x, y_ = _check_x(X)
        r = y
        x_z = x.dot(self.Z)
        preds = np.array([x_z[row].dot(y_[row]) for row in range(x.shape[0])])
        rmse = -root_mean_squared_error(r, preds)
        return -rmse
