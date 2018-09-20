import numpy as np
from .abstract_matrix import AbstractDesignMatrix

class DenseDesignMatrix(AbstractDesignMatrix):
    
    def __init__(self, X):
        """
        Params:
        ------
        X : numpy array
        order : str, {'row_major', 'col_major', None}
        """

        self.X = X
        self.shape = self.X.shape

    def dot(self, v):
        return self.X.dot(v)

    def Tdot(self, v):
        return self.X.T.dot(v)

    def compute_fisher_info(self, weight, diag_only=False):
        if diag_only:
            return np.sum(weight[:, np.newaxis] * self.X ** 2, 0)
        else:
            return self.X.T.dot(weight[:, np.newaxis] * self.X)

    def toarray(self):
        return self.X

    def extract_matrix(self, order=None):
        return self.X