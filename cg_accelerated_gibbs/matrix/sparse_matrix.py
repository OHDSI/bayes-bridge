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

    def matmul_by_diag(self, v, from_, order=None):

        v_mat = sparse.dia_matrix((v, 0), (len(v), len(v)))
        if from_ == 'left':
            if self.X_row_major is not None:
                X = self.X_row_major
            else:
                X = self.X
            X_multiplied = v_mat.dot(X)

        elif from_ == 'right':
            if self.X_col_major is not None:
                X = self.X_col_major
            else:
                X = self.X
            X_multiplied = X.dot(v_mat)

        return SparseMatrix(X_multiplied, order)

    def transpose(self):

        if self.order == 'row_major':
            X_T = SparseMatrix(self.X_row_major.T, order='col_major')
        elif self.order == 'col_major':
            X_T = SparseMatrix(self.X_col_major.T, order='row_major')
        else:
            X = (self.X_col_major.T, self.X_row_major.T)
            X_T = SparseMatrix(X)

        return X_T

    def matdot(self, another):

        if self.format == another.format == 'sparse':
            if self.order == 'col_major':
                A_row_major = self.X_col_major.csr()
            else:
                A_row_major = self.X_row_major
            if another.order == 'row_major':
                B_col_major = another.X_row_major.csc()
            else:
                B_col_major = another.X_col_major
            return A_row_major.dot(B_col_major).toarray()

        else:
            raise NotImplementedError()

    def sqnorm(self, axis=0):

        if axis != 0:
            raise NotImplementedError()

        if self.order == 'row_major':
            sq_norm = self.X_row_major.power(2).sum(0)
        else:
            sq_norm = self.X_col_major.power(2).sum(0)
        sq_norm = np.squeeze(np.asarray(sq_norm))

        return sq_norm

    def elemwise_power(self, exponent, order=None):

        if order == 'col_major' and (self.X_col_major is not None):
            X_powered = self.X_col_major.power(exponent)
        elif order == 'row_major' and (self.X_row_major is not None):
            X_powered = self.X_row_major.power(exponent)
        else:
            X_powered = self.X.power(exponent)

        return SparseMatrix(X_powered, order)

    def sum(self, axis):
        # TODO: optimally choose row or col major depending on the summation axis?
        return np.asarray(self.X.sum(axis=axis))

    def switch_order(self, target_order):

        if self.order is None:
            raise NotImplementedError()

        if self.order != target_order:
            if target_order == 'row_major':
                self.X_row_major = self.X_col_major.csr()
                self.X = self.X_row_major
                self.order = target_order
            elif target_order == 'col_major':
                self.X_col_major = self.X_row_major.csc()
                self.X = self.X_col_major
                self.order = target_order
            else:
                raise NotImplementedError()

    def toarray(self):
        return self.X.toarray()

    def extract_matrix(self, order=None):

        if order == 'row_major':
            if self.X_row_major is not None:
                X = self.X_row_major
            else:
                X = self.X.csr()
        elif order == 'col_major':
            if self.X_col_major is not None:
                X = self.X_col_major
            else:
                X = self.X.tocsc()
        else:
            X = self.X

        return X