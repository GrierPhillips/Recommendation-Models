import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_array


def root_mean_squared_error(true, pred):
    """Calculate the root mean sqaured error.

    Parameters
    ----------
        true : array, shape (n_samples)
            Array of true values.
        pred : array, shape (n_samples)
            Array of predicted values.
    Returns
    -------
        rmse : float
            Root mean squared error for the given values.

    """
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return rmse


def _check_x(X, indices=False):
    if indices:
        len_ = 4
    else:
        len_ = 2
    if isinstance(X, tuple):
        if len(X) != len_:
            raise ValueError('Argument X should be a tuple of length {}.'
                             .format(len_))
        out = X
    elif isinstance(X, DataHolder):
        if len_ == 2:
            out = (X.X, X.Y)
        else:
            out = (X.X, X.Y, X.X_ind, X.Y_ind)
    else:
        raise TypeError('Type of argument X should be tuple or DataHolder, was'
                        ' {}.'.format(str(type(X)).split("'")[1]))
    return out



    Parameters
    ----------



    Returns
    -------

    """
    return out


class DataHolder(object):
    """Class for packing user and item attributes into sigle data structure.

    Parameters
    ----------
    X : array-like, shape (n_samples, p_attributes)
        Array of user attributes. Each row represents a user.

    Y : array-like, shape (m_samples, q_attributes)
        Array of item attributes. Each row represents an item.

    """

    def __init__(self, X, Y, X_ind=None, Y_ind=None):
        """Initialize instance of DataHolder."""
        self.X = X
        self.Y = Y
        if X_ind is not None and Y_ind is not None:
            self.X_ind = X_ind
            self.Y_ind = Y_ind
        self.shape = self.X.shape

    def __getitem__(self, x):
        """Return a tuple of the requested index for both X and Y."""
        if hasattr(self, 'X_ind'):
            return self.X[x], self.Y[x], self.X_ind[x], self.Y_ind[x]
        return self.X[x], self.Y[x]
