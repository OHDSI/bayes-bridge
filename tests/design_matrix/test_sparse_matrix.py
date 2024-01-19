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
    weight = np.random.exponential(size=X.shape[1])
    return weight


def test_compute_transposed_fisher_info(X, X_sp, weight):
    expected_result = X @ np.diag(weight) @ X.T
    for add_intercept in [False, True]:
        design = SparseDesignMatrix(
            X_sp, center_predictor=False, add_intercept=add_intercept
        )
        assert np.allclose(
            design.compute_transposed_fisher_info(weight),
            expected_result
        )


def test_compute_transposed_fisher_info_centered(X, X_sp, weight):
    X_centered = X - X.mean(0)
    expected_result = X_centered @ np.diag(weight) @ X_centered.T
    for add_intercept in [False, True]:
        design = SparseDesignMatrix(
            X_sp, center_predictor=True, add_intercept=add_intercept
        )
        assert np.allclose(
            design.compute_transposed_fisher_info(weight),
            expected_result
        )
