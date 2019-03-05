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
        super().__init__(X)
        self.X = X

    def dot(self, v):

        if self.memoized and np.all(self.v_prev == v):
            return self.X_dot_v

        result = self.X.dot(v)
        if self.memoized:
            self.X_dot_v = result
            self.v_prev = v
        self.dot_count += 1

        return result

    def Tdot(self, v):
        self.Tdot_count += 1
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