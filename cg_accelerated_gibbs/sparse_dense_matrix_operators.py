import scipy as sp
import numpy as np


""" Basic matrix operations made to work for both sparse and dense matrices. """


def elemwise_power(A, exponent):
    if sp.sparse.issparse(A):
        return A.power(exponent)
    else:
        return A ** exponent


def left_matmul_by_diag(v, A):
    """ Computes dot(diag(v), A) for a vector 'v' and matrix 'A'. """
    if sp.sparse.issparse(A):
        v_mat = sp.sparse.dia_matrix((v, 0), (len(v), len(v)))
        return v_mat.dot(A)
    else:
        return v[:, np.newaxis] * A


def right_matmul_by_diag(A, v):
    """ Computes dot(A, diag(v)) for a matrix 'A' and vector 'v'. """
    if sp.sparse.issparse(A):
        v_mat = sp.sparse.dia_matrix((v, 0), (len(v), len(v)))
        return A.tocsc().dot(v_mat)
    else:
        return A * v[np.newaxis, :]


def choose_optimal_format_for_matvec(A_row_major, A_col_major):
    A = A_row_major
    if A_col_major is not None:
        A_T = A_col_major.T
    else:
        A_T = A_row_major.T
    return A, A_T