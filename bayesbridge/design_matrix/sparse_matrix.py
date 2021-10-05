from warnings import warn
import numpy as np
import scipy.sparse as sparse
from .abstract_matrix import AbstractDesignMatrix
try:
    from .mkl_matvec import mkl_csr_matvec
except:
    mkl_csr_matvec = None
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    cp = None
    cupy_exception = e


class SparseDesignMatrix(AbstractDesignMatrix):

    def __init__(self, X, use_mkl=True, center_predictor=False, add_intercept=True,
                 copy_array=False, dot_format='csr', Tdot_format='csr'):
        """
        Params:
        ------
        X : scipy sparse matrix
        """
        if copy_array:
            X = X.copy()
        super().__init__()
        if dot_format == 'csc' or Tdot_format == 'csc':
            raise NotImplementedError(
                "Current dot operations are only implemented for the CSR format."
            )
        self.use_cupy = self.is_cupy_sparse(X)
        if use_mkl and (mkl_csr_matvec is None) and (not self.use_cupy):
            warn("Could not load MKL Library. Will use Scipy's 'dot'.")
            use_mkl = False
        self.centered = center_predictor
        self.intercept_added = add_intercept
        self.use_mkl = (not self.use_cupy) and use_mkl
        X = self.remove_intercept_indicator(X)
        squeeze, array, zeros = (cp.squeeze, cp.array, cp.zeros) if self.use_cupy \
            else (np.squeeze, np.array, np.zeros)
        if center_predictor:
            self.column_offset = squeeze(array(X.mean(axis=0)))
        else:
            self.column_offset = zeros(X.shape[1])
        self.X_main = cp.sparse.csr_matrix(X) if self.use_cupy else X.tocsr()

    @property
    def shape(self):
        shape = self.X_main.shape
        return shape[0], shape[1] + int(self.intercept_added)

    @property
    def is_sparse(self):
        return True

    @property
    def nnz(self):
        """ Currently exists only to estimate the cost of matrix-matrix and
        matrix-vector operations. Does not correspond to the actual nnz of the
        represented design matrix.
        """
        return self.X_main.nnz

    def dot(self, v):
        if self.memoized:
            if np.all(self.v_prev == v):
                return self.X_dot_v
            self.v_prev = v.copy()

        input_is_cupy = (cp is not None) and isinstance(v, cp.ndarray)
        if self.use_cupy and not input_is_cupy:
            v = cp.asarray(v)
        intercept_effect = 0.
        if self.intercept_added:
            intercept_effect += v[0]
            v = v[1:]
        result = intercept_effect + self.main_dot(v)
        if self.use_cupy and not input_is_cupy:
            result = cp.asnumpy(result)
        if self.memoized:
            self.X_dot_v = result
        self.dot_count += 1

        return result

    def main_dot(self, v):
        """ Multiply by the main effect part of the design matrix. """
        X = self.X_main
        if self.use_mkl:
            result = mkl_csr_matvec(X, v)
        else:
            result = X.dot(v)
        inner = cp.inner if self.use_cupy else np.inner
        result -= inner(self.column_offset, v)
        if self.memoized:
            self.X_dot_v = result
        return result

    def Tdot(self, v):
        input_is_cupy = (cp is not None) and isinstance(v, cp.ndarray)
        if self.use_cupy and not input_is_cupy:
            v = cp.asarray(v)

        result = self.main_Tdot(v)

        if self.intercept_added:
            if self.use_cupy:
                result = cp.concatenate((cp.asarray([cp.sum(v)]), result))
            else:
                result = np.concatenate(([np.sum(v)], result))
        self.Tdot_count += 1

        if self.use_cupy and not input_is_cupy:
            result = cp.asnumpy(result)
        return result

    def main_Tdot(self, v):
        X = self.X_main
        if self.use_mkl:
            result = mkl_csr_matvec(X, v, transpose=True)
        else:
            result = X.T.dot(v)
        sum = cp.sum if self.use_cupy else np.sum
        result -= sum(v) * self.column_offset
        return result

    def compute_fisher_info(self, weight, diag_only=False):
        """ Compute $X^T W X$ where W is the diagonal matrix of a given weight."""

        if diag_only:
            return self.compute_fisher_diag(weight)

        weight_mat = self.create_diag_matrix(weight)
        X = self.X_main
        X_T = X.T
        weighted_X = weight_mat.dot(X).tocsc()

        n_pred = self.shape[1]
        fisher_info = np.zeros((n_pred, n_pred))
        if self.intercept_added:
            fisher_info[0, 0] = np.sum(weight)
            fisher_info[0, 1:] \
                = weighted_X.sum(0) - np.sum(weight) * self.column_offset
            fisher_info[1:, 0] = fisher_info[0, 1:]
            fisher_info_wo_intercept = fisher_info[1:, 1:]
        else:
            fisher_info_wo_intercept = fisher_info

        fisher_info_wo_intercept += X_T.dot(weighted_X).toarray()
        if self.centered:
            outer_prod_term = np.outer(
                self.column_offset, weighted_X.sum(0)
            )
            fisher_info_wo_intercept -= outer_prod_term + outer_prod_term.T
            fisher_info_wo_intercept \
                += np.sum(weight) * np.outer(self.column_offset, self.column_offset)

        return fisher_info

    def compute_fisher_diag(self, weight):

        weight_mat = self.create_diag_matrix(weight)
        diag = weight_mat.dot(self.X_main.power(2)).sum(0)
        if self.centered:
            weighted_X = weight_mat.dot(self.X_main).tocsc()
            diag -= 2 * self.column_offset \
                    * np.squeeze(np.asarray(weighted_X.sum(0)))
            diag += np.sum(weight) * self.column_offset ** 2
        diag = np.squeeze(np.asarray(diag))
        if self.intercept_added:
            diag = np.concatenate(([np.sum(weight)], diag))

        return diag

    def create_diag_matrix(self, v):
        return sparse.dia_matrix((v, 0), (len(v), len(v)))

    def toarray(self):
        X = self.X_main.toarray() - self.column_offset[np.newaxis, :]
        if self.intercept_added:
            X = np.hstack((np.ones(X.shape[0]), X))
        return X

    def extract_matrix(self, order=None):
        pass
