"""
Implementation of Dirty Inductive Matrix Completion.

Dirty Inductive Matrix Completion (DirtyIMC) combines Inductive Matrix
Completion (IMC) with other methods for solving matrix completion problems such
as Alternating Least Squares (ALS). This module implements DirtyIMC using IMC
described in P. Jain and I. S. Dhillon (2013) Provable Inductive Matrix
Completion arXiv:1306.0626, and ALS described in Zhou Y., Wilkinson D.,
Schreiber R., Pan R. (2008) Large-Scale Parallel Collaborative Filtering for
the Netflix Prize.

"""
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from . import ALS, IMC
from .utils import _check_x, _check_y, root_mean_squared_error


class DirtyIMC(BaseEstimator):
    """Implementation of Dirty Inductive Matrix Completion.

    Parameters
    ----------
    rank : integer (default=10)
        The number of latent features (rank) to include in the matrix
        factorization.

    n_components : int or None
        Number of components, if n_components is not set or is greater than the
        number of features in the item attributes matrix, all features are
        kept.

    method : None | 'Newton-CG' | 'BFGS' (default='Newton-CG')
        Algorithm used to find W and H that minimize the cost function.
        Default: 'BFGS'

    als_alpha : float, optional (default=0.1)
        Float representing the regularization penalty.

    imc_alpha : double, (default=0.1)
        Constant that multiplies the regularization terms. Set to zero to have
        no regularization.

    tol : float, optional (default=0.1)
        Float representing the difference in RMSE between iterations at which
        to stop factorization.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the ALS fitting process.

    Attributes
    ----------
    als : ALS
        ALS instance with given input parameters.

    imc : IMC
        IMC instance with given input parameters.

    reconstruction_err_ : float
        The sum squared error between the values predicted by the model and the
        real values of the training data.

    """

    def __init__(self, rank=10, n_components=None, method='Newton-CG',
                 als_alpha=0.1, imc_alpha=0.1, tol=0.1, random_state=None,
                 n_jobs=1, verbose=0):
        """Initialize instance of DirtyIMC."""
        self.als = ALS(rank=rank, alpha=als_alpha, tol=tol,
                       random_state=random_state, n_jobs=n_jobs,
                       verbose=verbose)
        self.imc = IMC(n_components=n_components, method=method,
                       alpha=imc_alpha, verbose=verbose)

    def fit(self, X, y, Z=None, shape=None):
        """Fit the model to the given data.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing arrays of user indices and item indices.

        y : {array-like, sparse matrix}
            1-D array or sparse matrix representing the data to be modeled.

        Z : array-like, shape (p_components, q_components)
            Initial guess for the IMC solution.

        shape : tuple or None, (default=None)
            If y is a 1-D array shape must be the shape of the real data.

        Returns
        -------
        self

        """
        _, _, _ = self.fit_transform(X, y, Z=Z, shape=shape)
        return self

    def fit_transform(self, X, y, Z=None, shape=None):
        """Learn DirtyIMC model for given data and return the decompositions.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing arrays of user indices and item indices.

        y : {array-like, sparse matrix}
            1-D array or sparse matrix representing the data to be modeled.

        Z : array-like, shape (p_attributes, q_attributes)
            Initial guess for the IMC solution.

        shape : tuple or None, (default=None)
            If y is a 1-D array shape must be the shape of the real data.

        Returns
        -------
        Z : array, shape (p_attributes, q_attributes)
            The IMC decomposition.

        user_feats : array, shape (k_components, n_samples)
            The array of latent user features.

        item_feats : array, shape (k_components, m_samples)
            The array of latent item features.

        """
        x, y_, users, items = _check_x(X, indices=True)
        if (y.ndim < 2 or y.shape[0] == 1) and not shape:
            raise ValueError('When y is a scalar or 1-D array shape must be' +
                             'provided.')
        r_ = _check_y(y, users, items)
        self.imc.fit((x, y_), r_, Z=Z)
        x_z = x.dot(self.imc.Z)
        xzy = (x_z * y_).sum(-1)
        if not shape:
            shape = y.shape
        self.als.fit((users, items), r_ - xzy, shape=shape)
        self.reconstruction_err_ = self.score(X, y)
        return self.imc.Z, self.als.user_feats, self.als.item_feats

    def predict(self, X):
        """Make predictions for the given data.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing arrays of user attributes, item attributes,
            user indices, and item indices.

        Returns
        -------
        prediction : array
            1-D array of all predicted values for the given user/item pairs.

        """
        check_is_fitted(self.imc, ['Z'])
        check_is_fitted(self.als, ['item_feats', 'user_feats'])
        x, y_, users, items = _check_x(X, indices=True)
        U = self.als.user_feats[:, users]
        V = self.als.item_feats[:, items]
        uv = (U.T * V.T).sum(-1)
        xzy = (x.dot(self.imc.Z) * y_).sum(-1)
        prediction = uv + xzy
        return prediction

    def score(self, X, y):
        """Return the root mean squared error for the predicted values.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing arrays of user attributes, item attributes,
            user indices, and item indices.

        y : {array-like, sparse matrix}
            The true values as a 1-D array or sparse matrix.

        Returns
        -------
        rmse : float
            The root mean squared error for the given the values.

        """
        if issparse(y):
            _, _, users, items = _check_x(X, indices=True)
            r_ = y[users, items].A1
        else:
            r_ = y
        preds = self.predict(X)
        rmse = -root_mean_squared_error(r_, preds)
        return rmse

    def update_user(self, user, item, value):
        """Update a single user's feature vector.

        When an existing user rates an item the feature vector for that user
        can be updated withot having to rebuild the entire model. Eventually,
        the entire model should be rebuilt, but this is as close to a real-time
        update as is possible.

        Parameters
        ----------
        user : integer
            Index for the user.

        item : integer
            Index for the item

        value : integer
            The value assigned to item by user.

        """
        self.als.update_user(user, item, value)

    def add_user(self):
        """Add a user to the model.

        When a new user is added append a new row to the data matrix and
        create a new column in user_feats. When the new user rates an item,
        the model will be ready insert the rating and use the update_user
        method to calculate the least squares approximation of the user
        features.

        """
        self.als.add_user()
