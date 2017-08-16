"""
Implementation of Inductive Matric Completion.

Inductive Matrix Completion (IMC) is an algorithm for recommender systems with
side-information of users and items. The IMC formulation incorporates features
associated with rows (users) and columns (items) in matrix completion, so that
it enables predictions for users or items that were not seen during training,
and for which only features are known but no dyadic information (such as
ratings or linkages).
"""


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
        pass

    def fit_transform(self, ratings, X, y):
        pass

    def fit(self, ratings, X, y):
        pass

    @staticmethod
    def _fv(V, *args):
        pass

    @staticmethod
    def _fv_prime(V, *args):
        pass

    @staticmethod
    def _fv_hess(V, p, *args):
        pass

    @staticmethod
    def _fu(U, *args):
        pass

    @staticmethod
    def _fu_prime(U, *args):
        pass

    @staticmethod
    def _fu_hess(U, p, *args):
        pass

    def predict_one(self):
        pass

    def predict_all(self):
        pass

    def score(self):
        pass

