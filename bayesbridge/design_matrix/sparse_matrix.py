import numpy as np
import scipy.sparse as sparse
from .abstract_matrix import AbstractDesignMatrix
from .mkl_matvec import mkl_csr_matvec


class SparseDesignMatrix(AbstractDesignMatrix):

    def __init__(self, X, use_mkl=True, centered=False,
                 dot_format='csr', Tdot_format='csr'):
        """
        Params:
        ------
        X_row_major : scipy sparse matrix
        """
        super().__init__(X)
        if dot_format == 'csc' or Tdot_format == 'csc':
            raise NotImplementedError(
                "Current dot operations are only implemented for the CSR format."
            )
        self.X_row_major = X.tocsr()
        self.use_mkl = use_mkl
        self.centered = centered
        if centered:
            self.X_colmean = np.squeeze(np.array(X.mean(axis=0)))
        else:
            self.X_colmean = np.zeros(self.X_row_major.shape[1])

    def dot(self, v):

        if self.memoized and np.all(self.v_prev == v):
            return self.X_dot_v

        X = self.X_row_major
        result = mkl_csr_matvec(X, v) \
            if self.use_mkl else X.dot(v)
        result -= np.inner(self.X_colmean, v)
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
        result -= self.X_colmean * np.sum(v)
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
        return self.X_row_major.toarray() - self.X_colmean[np.newaxis, :]

    def extract_matrix(self, order=None):
        pass