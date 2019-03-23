import numpy as np
import scipy.sparse as sparse
from .abstract_matrix import AbstractDesignMatrix
from .mkl_matvec import mkl_csr_matvec


class SparseDesignMatrix(AbstractDesignMatrix):

    def __init__(self, X, use_mkl=True, center_predictor=False, add_intercept=True,
                 dot_format='csr', Tdot_format='csr'):
        """
        Params:
        ------
        X : scipy sparse matrix
        """
        super().__init__(X)
        if dot_format == 'csc' or Tdot_format == 'csc':
            raise NotImplementedError(
                "Current dot operations are only implemented for the CSR format."
            )

        self.use_mkl = use_mkl

        self.centered = center_predictor
        if center_predictor:
            self.column_offset = np.squeeze(np.array(X.mean(axis=0)))
        else:
            self.column_offset = np.zeros(X.shape[1])

        if add_intercept:
            self.column_offset = np.concatenate(([0], self.column_offset))
            intercept_column = np.ones((X.shape[0], 1))
            X = sparse.hstack((intercept_column, X))

        self.X_row_major = X.tocsr()

    @property
    def shape(self):
        return self.X_row_major.shape

    def dot(self, v):

        if self.memoized and np.all(self.v_prev == v):
            return self.X_dot_v

        X = self.X_row_major
        result = mkl_csr_matvec(X, v) \
            if self.use_mkl else X.dot(v)
        result -= np.inner(self.column_offset, v)
        if self.memoized:
            self.X_dot_v = result
            self.v_prev = v
        self.dot_count += 1

        return result

    def Tdot(self, v):
        self.Tdot_count += 1
        X = self.X_row_major
        result = mkl_csr_matvec(X, v, transpose=True) \
            if self.use_mkl else X.T.dot(v)
        result -= self.column_offset * np.sum(v)
        return result

    def compute_fisher_info(self, weight, diag_only=False):

        if self.centered:
            raise NotImplementedError(
                "Operation not yet supported for a centered design.")

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
        return self.X_row_major.toarray() - self.column_offset[np.newaxis, :]

    def extract_matrix(self, order=None):
        pass