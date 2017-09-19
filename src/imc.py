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
from scipy.sparse import csc_matrix, diags, issparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted, check_array


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


def _fit_imc(R, X, Y, Z=None, method='BFGS', n_components=None,
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

    method : None | 'Newton-CG' | 'BFGS'
        Algorithm used to find W and H that minimize the cost function.
        Default: 'BFGS'

    n_components : int
        Number of components.

    alpha : double, default: 0.1
        Constant that multiplies the regularization terms.

    verbose : integer, default: 0
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


def _check_init(A, shape, whom):
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to {}. Expected {}, '
                         'but got {}.'.format(whom, shape, np.shape(A)))
    if np.max(A) == 0:
        raise ValueError('Array passed to {} is full of zeros.'.format(whom))


def _format_data(R, X, Y):
    """Ensure R, X, and Y are structured properly for IMC.

    The IMC is fit by utilizing a 1-d array of ratings, shape (n, ), and
    2-d arrays of user and item attributes, shape (n, p) and (n, q)
    respectively. This method ensures that when data is passed in as
    matrices of unique users and items or with a matrix of ratings, will be
    properly formatted.

    Parameters
    ----------
    R : {array-like, sparse matrix}, shape (n_samples, m_samples)
        Data matrix to be decomposed.

    X : array, shape (n_samples, p_attributes)
        Attribute matrix for users.

    Y : array, shape (m_samples, q_attributes)
        Attribute matrix for items.

    Returns
    -------
    r : array, shape (x_samples, )
        Array of actual ratings.

    x : array, shape (x_samples, p_attributes)
        Array of user attributes. The row index of the array corresponds to
        the index of the rating in r.

    y : array, shape (x_samples, q_attributes)
        Array of item attributes. The row index of the array corresponds to
        the index of the rating in r.

    """
    if R.ndim < 2 or R.shape[0] == 1:
        R = R.reshape(-1, 1)
    R = check_array(R, accept_sparse='csc')
    X = check_array(X, accept_sparse='csc')
    Y = check_array(Y, accept_sparse='csc')
    if not issparse(R):
        R = csc_matrix(R)
    rows, cols = R.nonzero()
    r = np.array(R[rows, cols]).flatten()
    x = X[rows]
    y = Y[rows]
    return r, x, y


def _check_x(X):
    if isinstance(X, tuple):
        if len(X) != 2:
            raise ValueError('Argument X should be a tuple of length 2 '
                             'containing an array for user attributes and an '
                             'array for item attributes.')
        Y = np.array(X[1])
        X = np.array(X[0])
    elif isinstance(X, DataHolder):
        Y = X.Y
        X = X.X
    else:
        raise TypeError('Type of argument X should be tuple or DataHolder, was'
                        ' {}.'.format(str(type(X)).split("'")[1]))
    if Y.ndim != 2 or X.ndim != 2:
        Y = Y.reshape(1, -1)
        X = X.reshape(1, -1)
    return X, Y


class DataHolder(object):
    """Class for packing user and item attributes into sigle data structure.

    Parameters
    ----------
    X : array-like, shape (n_samples, p_attributes)
        Array of user attributes. Each row represents a user.

    Y : array-like, shape (m_samples, q_attributes)
        Array of item attributes. Each row represents an item.

    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.shape = self.X.shape

    def __getitem__(self, x):
        return self.X[x], self.Y[x]


class IMC(BaseEstimator):
    """Implementation of Inductive Matrix Completion.

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set or is greater than the
        number of features in the item attributes matrix, all features are
        kept.

    method : None | 'Newton-CG' | 'BFGS'
        Algorithm used to find W and H that minimize the cost function.
        Default: 'BFGS'

    alpha : double, default: 0.1
        Constant that multiplies the regularization terms. Set to zero to have
        no regularization.

    l1_ratio : double, default: 0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    verbose : int, default: 0
        The verbosity level.

    Attributes
    ----------
    components_h : array, [n_components, q_attributes]
        H component of the data. This is the H matrix in the formula:
        ``R ~= XWHY``.

    components_w : array, [n_components, p_attributes]
        W component of the data. This is the W matrix in the formula:
        ``R ~= XWHY``.

    reconstruction_err_ : number
        Frobenius norm of the matrix difference between the training data and
        the reconstructed data from the fit produced by the model.
        ``|| R - XWHY ||_2``

    """

    def __init__(self, n_components=None, method='BFGS', alpha=0.1, l1_ratio=0,
                 verbose=0):
        """Initialize instance of IMC."""
        self.n_components = n_components
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    def fit_transform(self, X, y, Z=None):
        """Learn IMC model for given data and return the transformed data.

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
        X, Y = _check_x(X)
        R = y
        if self.n_components and self.n_components < X.shape[1]:
            self.n_components_ = self.n_components
        else:
            self.n_components_ = X.shape[1]
        r, x, y = _format_data(R, X, Y)
        Z, success, msg = _fit_imc(
            r, x, y, Z=Z, n_components=self.n_components_, method=self.method,
            alpha=self.alpha, verbose=self.verbose)
        if not success:
            warnings.warn(msg, ConvergenceWarning, stacklevel=1)
        self.Z = Z
        self.reconstruction_err_ = self.score((X, Y), R)
        return Z

    def fit(self, X, y):
        """Learn IMC model for the data R, with attribute data X and Y.

        Parameters
        ----------
        X : tuple, len = 2
            Tuple containing matrices of user attributes and item attributes.

        y : {array-like, sparse matrix}, shape (n_samples, m_samples)
            Data matrix to be decomposed.

        Returns
        -------
        self

        """
        X, Y = _check_x(X)
        self.fit_transform((X, Y), y)
        return self

    def _predict(self, X, Y):
        """Make predictions for the given attribute arrays.

        Parameters
        ----------
        X : array-like, shape (n_samples, p_attributes)
            Array of user attributes. Each row represents a user.

        Y : array-like, shape (m_samples, q_attributes)
            Array of item attributes. Each row represents an item.

        Returns
        -------
        prediction : {float, array} (n_samples, m_samples)
            Array of predicted values for the user/items pairs.

        """
        X = check_array(X, accept_sparse='csr')
        Y = check_array(Y, accept_sparse='csr')
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
        X, Y = _check_x(X)
        prediction = self._predict(X, Y)
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
        X, Y = _check_x(X)
        predictions = self._predict(X, Y)
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
        X, Y = _check_x(X)
        r, x, y = _format_data(y, X, Y)
        x_z = x.dot(self.Z)
        preds = np.array([x_z[row].dot(y[row].T) for row in range(x.shape[0])])
        mse = mean_squared_error(r, preds)
        rmse = np.sqrt(mse)
        return -rmse
