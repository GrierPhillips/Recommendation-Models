"""
Implementation of alternating least squares with regularization.

The alternating least squares with regularization algorithm ALS-WR was first
demonstrated in the paper Large-scale Parallel Collaborative Filtering for
the Netflix Prize. The authors discuss the method as well as how they
parallelized the algorithm in Matlab. This module implements the algorithm in
parallel in python with the built in concurrent.futures module.
"""

import os
import pickle
import subprocess

import numpy as np
import scipy.sparse as sps
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_random_state

from .utils import _format_data, root_mean_squared_error

# pylint: disable=E1101,W0212


class ALS(BaseEstimator):
    """Implementation of Alternative Least Squares for Matrix Factorization.

    Parameters
    ----------
    rank : integer (default=10)
        The number of latent features (rank) to include in the matrix
        factorization.

    alpha : float, optional (default=0.1)
        Float representing the regularization penalty.

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
    data : {array-like, sparse matrix} shape (n_samples, m_samples)
        Constant matrix representing the data to be modeled.

    item_features : array-like, shape (k_features, m_samples)
        Array of shape (rank, m_samples) where m represents the number of items
        contained in the data. Contains the latent features of items extracted
        by the factorization process.

    user_features : array-like, shape (k_features, n_samples)
        Array of shape (rank, n_samples) where n represents the number of users
        contained in the data. Contains the latent features of users extracted
        by the factorization process.

    reconstruction_err_ : float
        The sum squared error between the values predicted by the model and the
        real values of the training data.

    """

    def __init__(self, rank=10, alpha=0.1, tol=0.001, random_state=None,
                 n_jobs=1, verbose=0):
        """Initialize instance of ALS."""
        self.rank = rank
        self.alpha = alpha
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, shape=None):
        """Fit the model to the given data.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing arrays of user indices and item indices.

        y : {array-like, sparse matrix}
            1-D array or sparse matrix representing the data to be modeled.

        shape : tuple or None, (default=None)
            If y is a 1-D array shape must be the shape of the real data.

        Returns
        -------
        self

        """
        _, _ = self.fit_transform(X, y, shape=shape)
        return self

    def fit_transform(self, X, y, shape=None):
        """Fit the model to the given data.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing arrays of user indices and item indices.

        y : {array-like, sparse matrix}
            1-D array or sparse matrix representing the data to be modeled.

        shape : tuple or None, (default=None)
            If y is a 1-D array shape must be the shape of the real data.

        Returns
        -------
        user_feats : array, shape (k_components, n_samples)
            The array of latent user features.

        item_feats : array, shape (k_components, m_samples)
            The array of latent item features.

        """
        if (y.ndim < 2 or y.shape[0] == 1) and not shape:
            raise ValueError('When y is a scalar or 1-D array shape must be' +
                             'provided.')
        users, items, vals = _format_data(X, y)
        if not sps.issparse(y):
            data = sps.lil_matrix(shape)
            for idx, (i, j) in enumerate(zip(users, items)):
                data[i, j] = vals[idx]
            data = data.tocsr()
        else:
            data = y.tocsr()
        random_state = check_random_state(self.random_state)
        with open('random.pkl', 'wb') as state:
            pickle.dump(random_state.get_state(), state)
        sps.save_npz('data', data)
        try:
            subprocess.run(
                ['python', 'fit_als.py', str(self.rank), str(self.tol),
                 str(self.alpha), '-rs', 'random.pkl', '-j',
                 str(self.n_jobs), '-v', str(self.verbose)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as err:
            err_msg = '\n\t'.join(err.stderr.decode().split('\n'))
            raise ValueError('Fitting ALS failed with error:\n\t{}'
                             .format(err_msg))
        with np.load('features.npz') as loader:
            self.user_feats = loader['user']
            self.item_feats = loader['item']
        for _file in ['data.npz', 'features.npz', 'random.pkl']:
            os.remove(_file)
        self.data = data
        self.reconstruction_err_ = self.score(X, y)
        return self.user_feats, self.item_feats

    def _predict(self, X):
        """Make predictions for the given arrays.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing arrays of user indices and item indices.

        Returns
        -------
        predictions : array, shape (n_samples, m_samples)
            Array of all predicted values for the given user/item pairs.

        """
        check_is_fitted(self, ['item_feats', 'user_feats'])
        users, items = _format_data(X)
        U = self.user_feats.T[users]
        V = self.item_feats.T[items]
        predictions = (U * V).sum(-1)
        return predictions

    def predict_one(self, user, item):
        """Given a user and item provide the predicted rating.

        Predicted values for a single user, item pair can be provided by the
        fitted model by taking the dot product of the user column from the
        user_features and the item column from the item_features.

        Parameters
        ----------
        user : integer
            Index for the user.

        item : integer
            Index for the item.

        Returns
        -------
        prediction : float
            Predicted value at index user, item in original data.

        """
        prediction = self._predict((np.array([user]), np.array([item])))
        return prediction

    def predict_all(self, user):
        """Given a user provide all of the predicted values.

        Parameters
        ----------
        user : integer
            Index for the user.

        Returns
        -------
        predictions : array-like, shape (1, m_samples)
            Array containing predicted values of all items for the given user.

        """
        users = np.repeat(user, self.data.shape[1])
        items = np.arange(self.data.shape[1])
        predictions = self._predict((users, items))
        return predictions

    def score(self, X, y):
        """Return the root mean squared error for the predicted values.

        Parameters
        ----------
        X : tuple, DataHolder
            Structure containing row and column values for predictions.
        y : {array-like, sparse matrix}
            The true values as a 1-D array or stored in a sparse matrix.

        Returns
        -------
        rmse : float
            The root mean squared error for the test set given the values
            predicted by the model.

        """
        check_is_fitted(self, ['item_feats', 'user_feats'])
        users, items, y = _format_data(X, y)
        pred = np.array([
            self.user_feats[:, users[i]].T.dot(self.item_feats[:, items[i]])
            for i in range(users.shape[0])])
        rmse = -root_mean_squared_error(y, pred)
        return rmse

    def update_user(self, user, item, rating):
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

        rating : integer
            The value assigned to item by user.

        """
        check_is_fitted(self, ['item_feats', 'user_feats'])
        self.data[user, item] = rating
        submat = self.item_feats[:, self.data[user].indices]
        row = self.data[user].data
        col = self._update_one(submat, row, self.rank, self.lambda_)
        self.user_feats[:, user] = col

    def add_user(self, user_id):
        """Add a user to the model.

        When a new user is added append a new row to the data matrix and
        create a new column in user_feats. When the new user rates an item,
        the model will be ready insert the rating and use the update_user
        method to calculate the least squares approximation of the user
        features.

        Parameters
        ----------
        user_id : integer
            The index for the user.

        """
        check_is_fitted(self, ['item_feats', 'user_feats'])
        shape = self.data._shape
        if user_id >= shape[0]:
            self.data = sps.vstack([self.data, sps.csr_matrix((1, shape[1]))],
                                   format='csr')
        if user_id >= self.user_feats.shape[1]:
            new_col = np.zeros((self.rank, 1))
            self.user_feats = np.hstack((self.user_feats, new_col))
