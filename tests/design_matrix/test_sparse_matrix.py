import numpy as np
import pytest
import scipy as sp

from bayesbridge.design_matrix import SparseDesignMatrix


@pytest.fixture()
def X():
    np.random.seed(0)
    X = np.random.normal(size=(2, 4))
    return X


@pytest.fixture()
def X_sp(X):
    return sp.sparse.csr_matrix(X)


@pytest.fixture()
def weight(X):
    np.random.seed(0)
    weight = np.random.exponential(size=X.shape[1] + 1)
    return weight


def test_compute_transposed_fisher_info(X, X_sp, weight):
    design = SparseDesignMatrix(
        X_sp, center_predictor=False, add_intercept=True
    )
    assert np.allclose(
        design.compute_transposed_fisher_info(weight[1:] , include_intrcpt=False),
        X @ np.diag(weight[1:]) @ X.T
    )
    intrcpt_column = np.ones((X.shape[0], 1))
    X_with_intrcpt = np.hstack((intrcpt_column, X))
    assert np.allclose(
        design.compute_transposed_fisher_info(weight, include_intrcpt=True),
        X_with_intrcpt @ np.diag(weight) @ X_with_intrcpt.T
    )


def test_compute_transposed_fisher_info_centered(X, X_sp, weight):
    design = SparseDesignMatrix(
        X_sp, center_predictor=True, add_intercept=True
    )
    X_centered = X - X.mean(0)
    assert np.allclose(
        design.compute_transposed_fisher_info(weight[1:], include_intrcpt=False),
        X_centered @ np.diag(weight[1:]) @ X_centered.T
    )
    intrcpt_column = np.ones((X.shape[0], 1))
    X_centered_with_intrcpt = np.hstack((intrcpt_column, X_centered))
    assert np.allclose(
        design.compute_transposed_fisher_info(weight, include_intrcpt=True),
        X_centered_with_intrcpt @ np.diag(weight) @ X_centered_with_intrcpt.T
    )
