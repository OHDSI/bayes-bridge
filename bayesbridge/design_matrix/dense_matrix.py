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
        same_input = False
        if self.memoized:
            same_input = np.all(self.v_prev == v)
            if same_input:
                result = self.X_dot_v
            else:
                result = self.X.dot(v)
                self.X_dot_v = result
            self.v_prev = v
        else:
            result = self.X.dot(v)
        super().dot(None, same_input)
        return result

    def Tdot(self, v):
        super().Tdot(None)
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