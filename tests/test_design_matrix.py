import numpy as np

from bayesbridge.design_matrix import SparseDesignMatrix, DenseDesignMatrix
from simulate_data import simulate_design

atol = 10e-10
rtol = 10e-10


def test_sparse_design_intercept_and_centering():

    n_obs, n_pred = (100, 10)
    X = simulate_design(n_obs, n_pred, binary_frac=.5, format_='sparse')
    X_design = SparseDesignMatrix(X, centered=True, add_intercept=True)
    X_ndarray = center_and_add_intercept(X.toarray())
    w, v = (np.random.randn(size) for size in X_design.shape)
    assert np.allclose(
        X_design.dot(v), X_ndarray.dot(v), atol=atol, rtol=rtol
    )
    assert np.allclose(
        X_design.Tdot(w), X_ndarray.T.dot(w), atol=atol, rtol=rtol
    )


def test_dense_design_intercept_and_centering():
    n_obs, n_pred = (100, 10)
    X = simulate_design(n_obs, n_pred, binary_frac=.5, format_='dense')
    X_design = DenseDesignMatrix(X, centered=True, add_intercept=True)
    X_ndarray = center_and_add_intercept(X)
    w, v = (np.random.randn(size) for size in X_design.shape)
    assert np.allclose(
        X_design.dot(v), X_ndarray.dot(v), atol=atol, rtol=rtol
    )
    assert np.allclose(
        X_design.Tdot(w), X_ndarray.T.dot(w), atol=atol, rtol=rtol
    )


def center_and_add_intercept(X):
    X -= X.mean(axis=0)[np.newaxis, :]
    intercept_column = np.ones((X.shape[0], 1))
    X = np.hstack((intercept_column, X))
    return X