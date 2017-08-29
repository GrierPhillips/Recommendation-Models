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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted, check_array


INTEGER_TYPES = (numbers.Integral, np.integer)


def _cost(arr, *args):
    """Return the regularized sum squared error for given variables.

    Parameters
    ----------
    arr : array-like, shape (n_samples, )
        Concatenation of flattened components to minimize.

    Returns
    -------
    sse : The sum squared error for the given variables.

    """
    users, items, ratings, lam, l1_ratio, h_size, h_shape, w_shape = args
    h_component = arr[:h_size].reshape(h_shape)
    w_component = arr[h_size:].reshape(w_shape)
    x_m = users.dot(w_component.T).dot(h_component)
    preds = np.array([items[i].dot(x_m[i]) for i in range(users.shape[0])])
    sse = 0.5 * np.sum(
        (ratings - preds.flatten()) ** 2 +
        (1 - l1_ratio) * lam * (norm(w_component) + norm(h_component)) +
        2 * l1_ratio * lam * (norm(w_component, 1) + norm(h_component, 1))
    )
    return sse


def _cost_prime(arr, *args):
    """Return the gradient of the regularized SSE for given variables.

    Parameters
    ----------
    arr: array-like, shape (n_samples, )
        Concatenation of flattened components to minimize.

    Returns
    -------
    The gradient of the regularized sum squared error for the given variables.

    """
    users, items, ratings, lam, l1_ratio, h_size, h_shape, w_shape = args
    h_component = arr[:h_size].reshape(h_shape)
    w_component = arr[h_size:].reshape(w_shape)
    w_h = w_component.T.dot(h_component)

    def _h_prime():
        y_m = items.dot(w_h.T)
        h_diag = np.array([
            users[i].dot(y_m[i]) for i in range(users.shape[0])]).flatten()
        x_w = users.dot(w_component.T).T
        wxr = np.vstack([x_w[i] * ratings for i in range(x_w.shape[0])])
        g_h = items.T.dot(diags(h_diag).T.dot(x_w.T)) - items.T.dot(wxr.T) +\
            (1 - l1_ratio) * lam * h_component.T + 0.5 * l1_ratio * lam
        return g_h.T

    def _w_prime():
        x_m = users.dot(w_h)
        w_diag = np.array([
            items[i].dot(x_m[i]) for i in range(users.shape[0])]).flatten()
        y_h = items.dot(h_component.T).T
        hyr = np.vstack([y_h[i] * ratings for i in range(y_h.shape[0])])
        g_w = users.T.dot(diags(w_diag).T.dot(y_h.T)) - users.T.dot(hyr.T) +\
            (1 - l1_ratio) * lam * w_component.T + 0.5 * l1_ratio * lam
        return g_w.T

    return np.concatenate([_h_prime().flatten(), _w_prime().flatten()])


def _cost_hess(arr, s_vec, *args):
    """Return the hessian of the regularized SSE for given variables.

    Parameters
    ----------
    arr : array-like, shape (n_samples, )
        Concatenation of flattened components to minimize.

    s_vec : array-like, shape (n_samples, )
        Concatenation of arbitrary vectors used to multiply the hessian.

    Returns
    -------
    The hessian of the regularized sum squared error for the given variables.

    """
    users, items, _, lam, l1_ratio, h_size, h_shape, w_shape = args
    h_component = arr[:h_size].reshape(h_shape)
    w_component = arr[h_size:].reshape(w_shape)

    def _h_hess():
        s_h = s_vec[:h_size].reshape(h_shape)
        w_s = w_component.T.dot(s_h)
        h_diag = np.array([
            items[i].dot(users[i].dot(w_s).T).T
            for i in range(users.shape[0])]).flatten()
        g_h = items.T.dot(diags(h_diag).T.dot(users).dot(w_component.T)).T +\
            (1 - l1_ratio) * lam * s_h
        return g_h

    def _w_hess():
        s_w = s_vec[h_size:].reshape(w_shape)
        h_s = h_component.T.dot(s_w)
        w_diag = np.array([
            users[i].dot(items[i].dot(h_s).T).T
            for i in range(users.shape[0])]).flatten()
        g_w = users.T.dot(diags(w_diag).T.dot(items).dot(h_component.T)).T +\
            (1 - l1_ratio) * lam * s_w
        return g_w

    return np.concatenate([_h_hess().flatten(), _w_hess().flatten()])


def _fw(arr, *args):
    """Return the hessian of the regularized SSE for a given W.

    Parameters
    ----------
    arr: array-like, shape (n_samples, )
        Component to minimize the cost function against.

    Returns
    -------
    g_h: The sum squared error for the given H.

    """
    h_component, users, items, ratings, lam, l1_ratio, shape = args
    w_component = arr.reshape(shape)
    w_h = w_component.T.dot(h_component)
    x_m = users.dot(w_h)
    preds = np.array([items[i].dot(x_m[i]).T for i in range(users.shape[0])])
    g_w = 0.5 * np.sum(
        (ratings - preds.flatten()) ** 2 +
        (1 - l1_ratio) * lam * (norm(w_component) + norm(h_component)) +
        2 * l1_ratio * lam * (norm(w_component, 1) + norm(h_component, 1))
    )
    return g_w


def _fw_prime(arr, *args):
    """Return the hessian of the regularized SSE for a given array.

    Parameters
    ----------
    arr: array-like, shape (n_samples, n_features)
        Component to minimize the cost function against.

    Returns
    -------
    g_w: The gradient of the regularized sum squared error for the given array.

    """
    h_component, users, items, ratings, lam, l1_ratio, shape = args
    w_component = arr.reshape(shape)
    y_m = items.dot(w_component.T.dot(h_component).T)
    diag = np.array([
        users[i].dot(y_m[i]) for i in range(users.shape[0])]).flatten()
    y_h = items.dot(h_component.T).T
    hyr = np.vstack([y_h[i] * ratings for i in range(y_h.shape[0])])
    g_w = users.T.dot(diags(diag).T.dot(y_h.T)) - users.T.dot(hyr.T) +\
        (1 - l1_ratio) * lam * w_component.T + 0.5 * l1_ratio * lam
    return g_w.T.flatten()


def _fw_hess(_, s_vec, *args):
    """Return the hessian of the regularized SSE for a given array.

    Parameters
    ----------
    _ : array-like, shape (n_samples,)
        Component to minimize the cost function against.

    s_vec : array-like shape (n_samples, )
        Arbitrary vector to mulitply with hessian.

    Returns
    -------
    g_w: The hessian of the regularized sum squared error for the given array.

    """
    h_component, users, items, _, lam, l1_ratio, shape = args
    s_vec = s_vec.reshape(shape)
    h_s = h_component.T.dot(s_vec)
    yhs = items.dot(h_s)
    diag = np.array([
        users[i].dot(yhs[i].T).T for i in range(users.shape[0])]).flatten()
    g_w = users.T.dot(diags(diag).T.dot(items).dot(h_component.T)).T +\
        (1 - l1_ratio) * lam * s_vec
    return g_w.flatten()


def _fit_imc(R, X, Y, W=None, H=None, method='BFGS', n_components=None,
             alpha=0.1, l1_ratio=0, update_H=True, verbose=0):
    """Compute Inductive Matrix Completion (IMC) with AltMin-LRROM.

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is calculated by the Newton-CG algorithm.

    Parameters
    ----------
    R : {array-like, sparse matrix} shape (n_samples, m_samples)
        Constant matrix where rows correspond to users and columns to items.

    X : array-like, shape (n_samples, p_attributes)
        Constant user attribute matrix.

    Y : array-like, shape (m_samples, q_attributes)
        Constant item attribute matrix.

    W : array-like, shape (k_components, p_attributes)
        Initial guess for the solution.

    H : array-like, shape (k_components, q_attributes)
        Initial guess for the solution.

    method : None | 'Newton-CG' | 'BFGS'
        Algorithm used to find W and H that minimize the cost function.
        Default: 'BFGS'

    n_components : int
        Number of components.

    alpha : double, default: 0.1
        Constant that multiplies the regularization terms.

    l1_ratio : double, default: 0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an L2 penalty.
        For l1_ratio = 1 it is an L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : integer, default: 0
        The verbosity level.

    Returns
    -------
    W : array-like, shape (k_components, p_attributes)
        Solution to the least squares problem.

    H : array-like, shape (k_components, q_attributes)
        Solution to the least squares problem.

    res.success : boolean
        Whether or not the minimization converged.

    res.message :
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

    if H is None:
        H = np.zeros((n_components, Y.shape[1]))
    if W is None:
        sum_x_r_y = diags(R).T.dot(X).T.dot(Y) / R.size
        u, _, _ = svd(sum_x_r_y, False)
        W = u.T[:n_components]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        if update_H:
            x0 = np.concatenate([H.flatten(), W.flatten()])
            res = so.minimize(
                _cost, x0, method=method, jac=_cost_prime, hessp=_cost_hess,
                args=(X, Y, R, alpha, l1_ratio, H.size, H.shape, W.shape),
                options={'disp': verbose})
            H = res['x'][:H.size].reshape(H.shape)
            W = res['x'][H.size:H.size + W.size].reshape(W.shape)
        else:
            _check_init(H, (n_components, Y.shape[1]), 'IMC (Input H)')
            x0 = W.flatten()
            res = so.minimize(
                _fw, x0, args=(H, X, Y, R, alpha, l1_ratio, W.shape),
                method=method, jac=_fw_prime, hessp=_fw_hess,
                options={'disp': verbose})
            W = res['x'].reshape(W.shape)

    return W, H, res.success, res.message


def _check_init(A, shape, whom):
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to {}. Expected {}, '
                         'but got {}.'.format(whom, shape, np.shape(A)))
    if np.max(A) == 0:
        raise ValueError('Array passed to {} is full of zeros.'.format(whom))



    """
    H, X, Y, R, lam, l1_ratio, shape = args
    S = p.reshape(shape)
    gw = H.dot(Y.T).dot(Y).dot(H.T).dot(S).dot(X.T).dot(X) +\
        (1 - l1_ratio) * lam * S
    return gw.flatten()


class IMC(object):
    """Implementation of Inductive Matrix Completion.

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set or is greater than the
        number of features in the item attributes matrix, all features are
        kept.

    max_iter : int, default: 30
        Maximum number of iterations to compute.

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
    components_ : array, [n_components, n_features]
        Components of the data. This is the H matrix in the formula:
        ``R ~= XWHY``.

    reconstruction_err_ : number
        Frobenius norm of the matrix difference between the training data and
        the reconstructed data from the fit produced by the model.
        ``|| R - XWHY ||_2``

    """

    def __init__(self, n_components=None, max_iter=30, alpha=0.1, l1_ratio=0,
                 verbose=0):
        """Create instance of IMC with given parameters.

        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    def fit_transform(self, R, X, Y):
        """Learn IMC model for given data and return the transformed data.

        Parameters
        ----------
        R: {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed.
        X: array, shape (n_samples, p_features)
            Attribute matrix for users.
        Y: array, shape (n_features, q_features)
            Attribute matrix for items.

        Returns
        -------
        W: array, shape (min(self.n_components, q_features), p_features)

        """
        if not issparse(R):
            R = lil_matrix(R)
        sum_x_r_y = R.T.dot(X).T.dot(Y) / R.nonzero()[0].size
        u, _, _ = svd(sum_x_r_y, False)
        W = u.T
        if self.n_components and self.n_components < W.shape[0]:
            W = W[:self.n_components, :]
            self.n_components_ = self.n_components
        else:
            self.n_components_ = W.shape[0]
        H = np.random.rand(self.n_components_, Y.shape[1])
        W, H, _ = _fit_inductive_matrix_completion(
            R, X, Y, W, H, tol=1e-4, max_iter=self.max_iter, alpha=self.alpha,
            l1_ratio=self.l1_ratio, update_H=True, verbose=self.verbose)
        # for _ in range(self.max_iter):
        #     H_hat = so.fmin_ncg(
        #         self._fh,
        #         H.flatten(),
        #         self._fh_prime,
        #         fhess_p=self._fh_hess,
        #         args=(W, X, Y, R, self.alpha, self.l1_ratio, H.shape),
        #         disp=0
        #     ).reshape(H.shape)
        #     qH, _ = qr(H_hat.T)
        #     H = qH.T
        #     W_hat = so.fmin_ncg(
        #         self._fw,
        #         W.flatten(),
        #         self._fw_prime,
        #         fhess_p=self._fw_hess,
        #         args=(H, X, Y, R, self.alpha, self.l1_ratio, W.shape),
        #         disp=0
        #     ).reshape(W.shape)
        #     qW, _ = qr(W_hat.T)
        #     W = qW.T
        self.components_ = H
        return W

    def fit(self, R, X, Y):
        """Learn IMC model for the data R, with attribute data X and Y.

        Parameters
        ----------
        R: {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed.
        X: array, shape (n_samples, p_features)
            Attribute matrix for users.
        Y: array, shape (n_features, q_features)
            Attribute matrix for items.

        Returns
        -------
        self
        """
        self.fit_transform(R, X, Y)
        return self

    def transform(self, R):
        """Transform the data R according to the fitted IMC model.

        Parameters
        ----------
        R: {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be transformed by the model

        Returns
        -------
        W: array, shape (n_components, p_features)
            Transformed data. The W component of ``R = XWHY``.
        """
        check_is_fitted(self, 'n_components_')



    def predict_one(self):
        pass

    def predict_all(self):
        pass

    def score(self):
        pass

