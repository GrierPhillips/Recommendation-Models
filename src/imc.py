"""
Implementation of Inductive Matric Completion.

Inductive Matrix Completion (IMC) is an algorithm for recommender systems with
side-information of users and items. The IMC formulation incorporates features
associated with rows (users) and columns (items) in matrix completion, so that
it enables predictions for users or items that were not seen during training,
and for which only features are known but no dyadic information (such as
ratings or linkages).
"""

import numpy as np
from numpy.linalg import norm
import scipy.optimize as so


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

def _fw(W, *args):
    """Return the hessian of the regularized SSE for a given W.

    Parameters
    ----------
    W: array-like, shape (n_samples, n_features)
        Component to minimize the cost function against.

    Returns
    -------
    gw: The sum squared error for the given W.

    """
    H, X, Y, R, lam, l1_ratio, shape = args
    W = W.reshape(shape)
    gw = np.sum(
        0.5 * np.asarray(R - X.dot(W.T).dot(H).dot(Y.T)) ** 2 + 0.5 * lam *
        (1 - l1_ratio) * (norm(W) + norm(H)) + lam * l1_ratio *
        (norm(W, 1) + norm(H, 1))
    )
    return gw


def _fw_prime(W, *args):
    """Return the hessian of the regularized SSE for a given W.

    Parameters
    ----------
    W: array-like, shape (n_samples, n_features)
        Component to minimize the cost function against.

    Returns
    -------
    gw: The gradient of the regularized sum squared error for the given W.

    """
    H, X, Y, R, lam, l1_ratio, shape = args
    W = W.reshape(shape)
    gw = H.dot(Y.T).dot(Y).dot(H.T).dot(W).dot(X.T).dot(X) -\
        R.dot(Y).dot(H.T).T.dot(X) + lam * (1 - l1_ratio) * W +\
        l1_ratio * lam
    return gw.flatten()


def _fw_hess(W, p, *args):
    """Return the hessian of the regularized SSE for a given W.

    Parameters
    ----------
    W: array-like, shape (n_samples, n_features)
        Component to minimize the cost function against.

    Returns
    -------
    gw: The hessian of the regularized sum squared error for the given W.

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

