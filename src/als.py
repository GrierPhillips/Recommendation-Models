"""
Implementation of alternating least squares with regularization.

The alternating least squares with regularization algorithm ALS-WR was first
demonstrated in the paper Large-scale Parallel Collaborative Filtering for
the Netflix Prize. The authors discuss the method as well as how they
parallelized the algorithm in Matlab. This module implements the algorithm in
parallel in python with the built in concurrent.futures module.
"""

import os
import subprocess

from joblib import Parallel, delayed
import numpy as np
import scipy.sparse as sps
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_random_state

from .utils import _check_x, _check_y, root_mean_squared_error

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
        if n_jobs == -1:
            n_jobs = os.cpu_count()
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
        users, items = _check_x(X)
        if not sps.issparse(y):
            data = sps.lil_matrix(shape)
            for idx, (i, j) in enumerate(zip(users, items)):
                data[i, j] = y[idx]
            data = data.tocsr()
        else:
            data = y.tocsr()
        random_state = check_random_state(self.random_state)

        rmse = float('inf')
        diff = rmse
        item_avg = data.sum(0) / (data != 0).sum(0)
        item_avg[np.isnan(item_avg)] = 0
        self.item_feats = random_state.rand(self.rank, data.shape[1])
        self.item_feats[0] = item_avg
        self.user_feats = np.zeros((self.rank, data.shape[0]))
        self.data = data

        while diff > self.tol:
            user_arrays = np.array_split(np.arange(self.data.shape[0]),
                                         self.n_jobs)
            self._update_parallel(user_arrays)
            item_arrays = np.array_split(np.arange(self.data.shape[1]),
                                         self.n_jobs)
            self._update_parallel(item_arrays, user=False)
            users, items = data.nonzero()
            U = self.user_feats.T[users]
            V = self.item_feats.T[items]
            pred = (U * V).sum(-1)
            new_rmse = root_mean_squared_error(data.data, pred)
            diff = rmse - new_rmse
            rmse = new_rmse
            users, items = data.nonzero()
            self.reconstruction_err_ = self.score(X, y)

        return self.user_feats, self.item_feats

    def _update_parallel(self, arrays, user=True):
        """Update the given features in parallel.

        Parameters
        ----------
        arrays : ndarray
            Array of indices that represent which column of the features is
            being updated.
        user : bool
            Boolean indicating wheter or not user features are being updated.

        """
        params = {'rank': self.rank, 'alpha': self.alpha, 'user': user}
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._thread_update_features)(array, params)
                for array in arrays)
        for result in out:
            for index, value in result.items():
                if user:
                    self.user_feats[:, index] = value
                else:
                    self.item_feats[:, index] = value

    def _thread_update_features(self, indices, params):
        """Split updates of feature matrices to multiple threads.

        Args:
            indices (np.ndarray): Array of integers representing the index of
                the user or item that is to be updated.
            params (dict): Parameters for the ALS algorithm.
        Returns:
            data (dict): Dictionary of data with the user or item to be updated
                as key and the array of features as the values.

        """
        data = {}
        out = Parallel(
            n_jobs=self.n_jobs, backend='threading')(
                delayed(self._update_one)(index, **params)
                for index in indices)
        for i, val in enumerate(out, start=indices[0]):
            data[i] = val
        return data

    def _update_one(self, index, **params):
        """Update a single column for one of the feature matrices.

        Parameters
        ----------
        index : int
            Integer representing the index of the user/item that is to be
            updated.
        params : dict
            Parameters for the ALS algorithm.

        Returns
        -------
        col : ndarray
            An array that represents a column from the feature matrix that is
            to be updated.

        """
        rank, alpha, user = params['rank'], params['alpha'], params['user']
        if user:
            submat = self.make_item_submats(index)
            row = self.data[index].data
        else:
            submat = self.make_user_submats(index)
            row = self.data[:, index].data
        num_ratings = row.size
        reg_sums = submat.dot(submat.T) + alpha * num_ratings * np.eye(rank)
        feature_sums = submat.dot(row[np.newaxis].T)
        try:
            col = np.linalg.inv(reg_sums).dot(feature_sums)
        except np.linalg.LinAlgError:
            col = np.zeros((1, rank))
        return col.ravel()

    def make_user_submats(self, item):
        """Get the user submatrix from a single item in the ratings matrix.

        Parameters
        ----------
        item : int
            Index of the item to construct the user submatrix for.

        Returns
        -------
        submat : np.ndarray
            Array containing the submatrix constructed by selecting the columns
            from the user features for the ratings that exist for the given
            column in the ratings matrix.

        """
        idx_dtype = sps.sputils.get_index_dtype(
            (self.data.indptr, self.data.indices),
            maxval=max(self.data.nnz, self.data.shape[0]))
        indptr = np.empty(self.data.shape[1] + 1, dtype=idx_dtype)
        indices = np.empty(self.data.nnz, dtype=idx_dtype)
        data = np.empty(self.data.nnz,
                        dtype=sps.sputils.upcast(self.data.dtype))
        sps._sparsetools.csr_tocsc(
            self.data.shape[0], self.data.shape[1],
            self.data.indptr.astype(idx_dtype),
            self.data.indices.astype(idx_dtype), self.data.data, indptr,
            indices, data)
        submat = self.user_feats[:, indices[indptr[item]:indptr[item + 1]]]
        return submat

    def make_item_submats(self, user):
        """Get the item submatrix from a single user in the ratings matrix.

        Parameters
        ----------
        user : int
            Index of the user to construct the user submatrix for.

        Returns
        -------
        submat : np.ndarray
            Array containing the submatrix constructed by selecting the columns
            from the item features for the ratings that exist for the given row
            in the ratings matrix.

        """
        submat = self.item_feats[:, self.data[user].indices]
        return submat

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
        users, items = _check_x(X)
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
        users, items = _check_x(X)
        r_ = _check_y(y, users, items)
        pred = (self.user_feats.T[users] * self.item_feats.T[items]).sum(-1)
        rmse = -root_mean_squared_error(r_, pred)
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
        check_is_fitted(self, ['item_feats', 'user_feats'])
        self.data[user, item] = value
        sps.save_npz('data', self.data)
        np.savez('features', user=self.user_feats, item=self.item_feats)
        subprocess.run(
            ['fit_als.py', '-r', str(self.rank), '-a', str(self.alpha),
             'One', str(user), 'data.npz', 'features.npz'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with np.load('feature.npz') as loader:
            user_feats = loader['user']
        self.user_feats[:, user] = user_feats
        for _file in ['data.npz', 'feature.npz']:
            os.remove(_file)

    def add_user(self):
        """Add a user to the model.

        When a new user is added append a new row to the data matrix and
        create a new column in user_feats. When the new user rates an item,
        the model will be ready insert the rating and use the update_user
        method to calculate the least squares approximation of the user
        features.

        """
        check_is_fitted(self, ['item_feats', 'user_feats'])
        shape = self.data._shape
        self.data = sps.vstack([self.data, sps.csr_matrix((1, shape[1]))],
                               format='csr')
        new_col = np.zeros((self.rank, 1))
        self.user_feats = np.hstack((self.user_feats, new_col))
