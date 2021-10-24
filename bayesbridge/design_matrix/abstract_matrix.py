import abc
import numpy as np
import scipy as sp
import scipy.sparse
import warnings
try:
    import cupy as cp
    import cupyx as cpx
except (ImportError, ModuleNotFoundError) as e:
    cp = None
    cupy_exception = e


class AbstractDesignMatrix():

    def __init__(self):
        self.dot_count = 0
        self.Tdot_count = 0
        self.memoized = False
        self.X_dot_v = None # For memoization
        self.v_prev = None # For memoization

    @property
    @abc.abstractmethod
    def shape(self):
        pass

    @abc.abstractmethod
    def dot(self, v):
        pass

    @abc.abstractmethod
    def Tdot(self, v):
        """ Multiply by the transpose of the matrix. """
        pass

    @property
    @abc.abstractmethod
    def is_sparse(self):
        pass

    def memoize_dot(self, flag=True):
        self.memoized = flag
        if self.v_prev is None:
            self.v_prev = np.full(self.shape[1], float('nan'))
        if not flag:
            self.X_dot_v = None
            self.v_prev = None

    @abc.abstractmethod
    def compute_fisher_info(self, weight, diag_only):
        """ Computes X' diag(weight) X and returns it as a numpy array. """
        pass

    @property
    def n_matvec(self):
        return self.dot_count + self.Tdot_count

    def get_dot_count(self):
        return self.dot_count, self.Tdot_count

    def reset_matvec_count(self, count=0):
        if not hasattr(count, "__len__"):
            count = 2 * [count]
        self.dot_count = count[0]
        self.Tdot_count = count[1]

    @abc.abstractmethod
    def toarray(self):
        """ Returns a 2-dimensional numpy array. """
        pass

    @staticmethod
    def is_cupy_matrix(X):
        return AbstractDesignMatrix.is_cupy_dense(X) \
            or AbstractDesignMatrix.is_cupy_sparse(X)

    @staticmethod
    def is_cupy_dense(X):
        return (cp is not None) and isinstance(X, cp.ndarray)

    @staticmethod
    def is_cupy_sparse(X):
        return (cp is not None) and isinstance(X, cpx.scipy.sparse.spmatrix)

    @staticmethod
    def remove_intercept_indicator(X):
        squeeze, array, power = (cp.squeeze, cp.array, cp.power) if \
            AbstractDesignMatrix.is_cupy_sparse(X) else (np.squeeze, np.array, np.power)
        if sp.sparse.issparse(X) or AbstractDesignMatrix.is_cupy_sparse(X):
            col_variance = squeeze(array(X.power(2).mean(axis=0) - power(X.mean(axis=0), 2)))
        else:
            col_variance = np.var(X, axis=0)
        has_zero_variance = (col_variance < X.shape[0] * 2 ** -52)
        if np.any(has_zero_variance):
            warnings.warn(
                "Intercept column (or numerically indistinguishable from "
                "such) detected. Do not add intercept manually. Removing...."
            )
            X = X[:, np.logical_not(has_zero_variance)]
        return X