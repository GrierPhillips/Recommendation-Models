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

    def __init__(self, n_components=None, max_iter=30, alpha=0.1, l1_ratio=0):
        """Create instance of IMC with given parameters.

        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit_transform(self, ratings, X, y):
        pass

    def fit(self, ratings, X, y):
        pass

    @staticmethod
    def _fv(V, *args):
        """Return the regularized sum squared error for a given V.

        Parameters
        ----------
        V: array-like, shape (n_samples, n_feature)
            Component to minimize the cost function against.

        Returns
        -------
        gv: The sum squared error for the given V.

        """
        U, X, Y, R, lam, l1_ratio, shape = args
        V = V.reshape(shape)
        gv = np.sum(
            0.5 * np.asarray(R - X.dot(U.T).dot(V).dot(Y.T)) ** 2 + 0.5 * lam *
            (1 - l1_ratio) * (norm(U) + norm(V)) + lam * l1_ratio *
            (norm(U, 1) + norm(V, 1))
        )
        return gv

    @staticmethod
    def _fv_prime(V, *args):
        """Return the gradient of the regularized SSE for a given V.

        Parameters
        ----------
        V: array-like, shape (n_samples, n_features)
            Component to minimize the cost function against.

        Returns
        -------
        gv: The gradient of the regularized sum squared error for the given V.

        """
        U, X, Y, R, lam, l1_ratio, shape = args
        V = V.reshape(shape)
        gv = U.dot(X.T).dot(X).dot(U.T).dot(V).dot(Y.T).dot(Y) -\
            R.T.dot(X).dot(U.T).T.dot(Y) + (1 - l1_ratio) * lam * V + 0.5 *\
            l1_ratio * lam
        return gv.flatten()

    @staticmethod
    def _fv_hess(V, p, *args):
        """Return the hessian of the regularized SSE for a given V.

        Parameters
        ----------
        V: array-like, shape (n_samples, n_features)
            Component to minimize the cost function against.

        Returns
        -------
        gv: The hessian of the regularized sum squared error for the given V.

        """
        U, X, Y, R, lam, l1_ratio, shape = args
        S = p.reshape(shape)
        gv = U.dot(X.T).dot(X).dot(U.T).dot(S).dot(Y.T).dot(Y) +\
            (1 - l1_ratio) * lam * S
        return gv.flatten()

    @staticmethod
    def _fu(U, *args):
        """Return the hessian of the regularized SSE for a given U.

        Parameters
        ----------
        U: array-like, shape (n_samples, n_features)
            Component to minimize the cost function against.

        Returns
        -------
        gu: The sum squared error for the given U.

        """
        V, X, Y, R, lam, l1_ratio, shape = args
        U = U.reshape(shape)
        gu = np.sum(
            0.5 * np.asarray(R - X.dot(U.T).dot(V).dot(Y.T)) ** 2 + 0.5 * lam *
            (1 - l1_ratio) * (norm(U) + norm(V)) + lam * l1_ratio *
            (norm(U, 1) + norm(V, 1))
        )
        return gu

    @staticmethod
    def _fu_prime(U, *args):
        """Return the hessian of the regularized SSE for a given U.

        Parameters
        ----------
        U: array-like, shape (n_samples, n_features)
            Component to minimize the cost function against.

        Returns
        -------
        gu: The gradient of the regularized sum squared error for the given U.

        """
        V, X, Y, R, lam, l1_ratio, shape = args
        U = U.reshape(shape)
        gu = V.dot(Y.T).dot(Y).dot(V.T).dot(U).dot(X.T).dot(X) -\
            R.dot(Y).dot(V.T).T.dot(X) + lam * (1 - l1_ratio) * U +\
            l1_ratio * lam
        return gu.flatten()

    @staticmethod
    def _fu_hess(U, p, *args):
        """Return the hessian of the regularized SSE for a given U.

        Parameters
        ----------
        U: array-like, shape (n_samples, n_features)
            Component to minimize the cost function against.

        Returns
        -------
        gu: The hessian of the regularized sum squared error for the given U.

        """
        V, X, Y, R, lam, l1_ratio, shape = args
        S = p.reshape(shape)
        gu = V.dot(Y.T).dot(Y).dot(V.T).dot(S).dot(X.T).dot(X) +\
            (1 - l1_ratio) * lam * S
        return gu.flatten()

    def transform(self):
        pass

    def predict_one(self):
        pass

    def predict_all(self):
        pass

    def score(self):
        pass

