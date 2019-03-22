import numpy as np

from bayesbridge.design_matrix import SparseDesignMatrix, DenseDesignMatrix
from simulate_data import simulate_design

atol = 10e-10
rtol = 10e-10


def test_sparse_design_centering():

    n_obs, n_pred = (100, 10)
    X = simulate_design(n_obs, n_pred, binary_frac=.5, format_='sparse')
    X_design = SparseDesignMatrix(X, centered=True, add_intercept=False)
    X_dense = X.toarray()
    X_dense -= X_dense.mean(axis=0)[np.newaxis, :]
    w, v = (np.random.randn(size) for size in X.shape)
    assert np.allclose(X_dense.mean(axis=0), np.zeros(X_dense.shape[1]))
    assert np.allclose(
        X_design.dot(v), X_dense.dot(v), atol=atol, rtol=rtol
    )
    assert np.allclose(
        X_design.Tdot(w), X_dense.T.dot(w), atol=atol, rtol=rtol
    )


def test_dense_design_centering():
    n_obs, n_pred = (100, 10)
    X = simulate_design(n_obs, n_pred, binary_frac=.5, format_='dense')
    X_design = DenseDesignMatrix(X, centered=True, add_intercept=False)
    X_centered = X_design.toarray()
    assert np.allclose(
        X_centered.mean(axis=0), np.zeros(X_centered.shape[1]),
        atol=atol, rtol=rtol
    )