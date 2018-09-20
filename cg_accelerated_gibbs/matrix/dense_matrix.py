import numpy as np
import scipy.sparse as sparse
from .abstract_matrix import AbstractMatrix

class DenseMatrix(AbstractMatrix):
    
    def __init__(self, X):
        """
        Params:
        ------
        X : numpy array
        order : str, {'row_major', 'col_major', None}
        """

        self.X = X
        self.format = 'dense'
        self.order = None
        self.shape = self.X.shape

    def dot(self, v):
        return self.X.dot(v)

    def Tdot(self, v):
        return self.X.T.dot(v)

    def matmul_by_diag(self, v, from_, order=None):
        """
        Computes dot(diag(v), X) if direc == 'left' or dot(X, diag(v))
        if direc == 'right'. Return a numpy or scipy.sparse array.

        Params:
        ------
        v : vector
        """
        if from_ == 'left':
            X_multiplied = v[:, np.newaxis] * self.X
        elif from_ == 'right':
            X_multiplied = self.X * v[np.newaxis, :]

        return DenseMatrix(X_multiplied, order)

    def transpose(self):
        return DenseMatrix(self.X.T)

    def matdot(self, another):
        return self.X.dot(another.X)

    def sqnorm(self, axis=0):
        return np.squeeze(np.sum(self.X ** 2, 0))

    def toarray(self):
        return self.X

    def extract_matrix(self, order=None):
        return self.X