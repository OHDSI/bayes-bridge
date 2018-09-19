import numpy as np
import scipy.sparse as sparse


class Matrix():
    """
    Implements basic matrix operations optimized for both sparse and dense
    matrices. Supports also operations useful for design matrices.
    """

    def __init__(self, X, order=None):
        """
        Params:
        ------
        X : numpy array, scipy sparse matrix, or tuple of (csr, csc) matrix
        order : str, {'row_major', 'col_major', None}
        """
        if type(X) is tuple:

            self.X_row_major, self.X_col_major = X
            self.X = self.X_row_major
            self.format = 'sparse'
            self.order = None

        elif sparse.issparse(X):

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

        else:

            self.X = X
            self.format = 'dense'
            self.order = None

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
        """
        Computes dot(diag(v), X) if direc == 'left' or dot(X, diag(v))
        if direc == 'right'. Return a numpy or scipy.sparse array.

        Params:
        ------
        v : vector
        """

        if self.format == 'sparse':

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

        else:
            if from_ == 'left':
                X_multiplied = v[:, np.newaxis] * self.X
            elif from_ == 'right':
                X_multiplied = self.X * v[np.newaxis, :]

        return Matrix(X_multiplied, order)

    def transpose(self):

        if self.format == 'sparse':
            if self.order == 'row_major':
                X_T = Matrix(self.X_row_major.T, order='col_major')
            elif self.order == 'col_major':
                X_T = Matrix(self.X_col_major.T, order='row_major')
            else:
                X = (self.X_col_major.T, self.X_row_major.T)
                X_T = Matrix(X)
        else:
            X_T = Matrix(self.X.T)

        return X_T

    def matdot(self, another):
        # Returns a numpy array.
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

        elif self.format == another.format == 'dense':
            return self.X.dot(another.X)

        else:
            raise NotImplementedError()

    def sqnorm(self, axis=0):

        if axis != 0:
            raise NotImplementedError()

        if self.format == 'sparse':
            if self.order == 'row_major':
                sq_norm = self.X_row_major.power(2).sum(0)
            else:
                sq_norm = self.X_col_major.power(2).sum(0)
        else:
            sq_norm = np.sum(self.X ** 2, 0)
        sq_norm = np.squeeze(np.asarray(sq_norm))

        return sq_norm

    def sum(self, axis):
        # TODO: optimally choose row or col major depending on the summation axis?
        return np.asarray(self.X.sum(axis=axis))

    def elemwise_power(self, exponent, order=None):

        if self.format == 'sparse':
            if order == 'col_major' and (self.X_col_major is not None):
                X_powered = self.X_col_major.power(exponent)
            elif order == 'row_major' and (self.X_row_major is not None):
                X_powered = self.X_row_major.power(exponent)
            else:
                X_powered = self.X.power(exponent)
        else:
            X_powered = self.X ** exponent

        return Matrix(X_powered, order)

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
        """ Returns a 2-dimensional numpy array. """
        if self.format == 'sparse':
            return self.X.toarray()
        else:
            return self.X

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