import numpy as np
import scipy.sparse as sparse
from .abstract_matrix import AbstractMatrix

class SparseMatrix(AbstractMatrix):

    def __init__(self, X, order=None):
        """
        Params:
        ------
        X : scipy sparse matrix, or tuple of (csr, csc) SparseMatrix
        order : str, {'row_major', 'col_major', None}
        """
        if type(X) is tuple:

            self.X_row_major, self.X_col_major = X
            self.X = self.X_row_major
            self.format = 'sparse'
            self.order = None

        else:

            if order == 'row_major':
                self.X_row_major = X.tocsr()
                self.X_col_major = None
                self.X = self.X_row_major
            elif order == 'col_major':
                self.X_row_major = None
                self.X_col_major = X.tocsc()
                self.X = self.X_col_major
            else:
                self.X_row_major = X.tocsr()
                self.X_col_major = X.tocsc()
                self.X = self.X_row_major
            self.format = 'sparse'
            self.order = order

        self.shape = self.X.shape

    def dot(self, v):
        if self.format == 'sparse' and (self.X_row_major is not None):
            return self.X_row_major.dot(v)
        else:
            return self.X.dot(v)

    def Tdot(self, v):
        if self.format == 'sparse' and (self.X_col_major is not None):
            return self.X_col_major.T.dot(v)
        else:
            return self.X.T.dot(v)

    def compute_fisher_info(self, weight, diag_only=False):

        weight_mat = self.create_diag_matrix(weight)

        if diag_only:
            diag = weight_mat.dot(self.X_row_major.power(2)).sum(0)
            return np.squeeze(np.asarray(diag))
        else:
            X_T = self.X_row_major.T
            weighted_X = weight_mat.dot(self.X_row_major).tocsc()
            return X_T.dot(weighted_X).toarray()

    def create_diag_matrix(self, v):
        return sparse.dia_matrix((v, 0), (len(v), len(v)))

    def toarray(self):
        return self.X.toarray()

    def extract_matrix(self, order=None):
        pass